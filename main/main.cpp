/* pico-faces on M5Stack Tab5（ESP32-P4）
 *
 * 上流 cpldcpu/pico-faces の C99 推論エンジン（upstream/engine）で 128×128 RGB の顔画像を
 * 生成し、Tab5 の 1280×720 画面に 5 倍で表示する。RP2350 版ファーム（firmware/main.c）の
 * 置き換えで、エンジンのソースは 1 バイトも変えていない。
 *
 * 構成:
 *   - モデル blob は app（.rodata）に埋め込み、起動時に PSRAM へコピーして読む
 *     （エンジンの重みステージングは既定の memcpy: PSRAM → 内部 SRAM の arena）
 *   - 2 コア並列は pf_par.c（rf_par_for の置き換え）。生成タスクはコア 0 固定
 *   - 操作: 画面タップ。USB シリアルからは上流と同じ `G <seed> [k] [class] [w]` も受ける
 *
 * USB シリアル（115200 / USB Serial-JTAG）:
 *   G <seed> [k_steps] [class] [w]   生成。class/w を省略すると上流の golden 規約
 *                                    （class = seed % n_cond, w = seed % (n_w+1) - 1）
 *   I                                モデル情報
 *   応答: OK seed=.. k=.. class=.. w=.. crc32=........ ms=..
 *   → ホストの rf_golden と同じ crc32 が出れば bit 一致（README 参照）
 */
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <M5Unified.h>

#include "driver/usb_serial_jtag.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_random.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"

extern "C" {
#include "rf_model.h"
#include "rf_ops.h"
void pf_par_init(void);
void pf_par_reset(void);
void pf_par_report(void);
#ifdef RF_PIE_P4
int pf_pie_selftest(void);   /* PIE カーネルを参照実装と照合（不一致なら PIE を切る） */
void pf_pie_bench(void);     /* カーネル別マイクロベンチ（ログのみ） */
void pf_pie_bench3(void);    /* PIE 命令のスループット */
int pf_stage_prepare(const rf_model_t *m); /* 密な重みの転置コピー（PSRAM）を作る */
extern int64_t pf_vae_us;    /* 直近の VAE decode の所要時間（pf_pie_decode.c） */
extern int pf_vae_pie;
#endif
/* dit.c の進捗フック（weak）と進捗変数 */
extern volatile uint8_t rf_progress;
extern volatile uint8_t rf_progress_total;
}

static const char *TAG = "pico_faces";

extern const uint8_t model_bin_start[] asm("_binary_model_bin_start");
extern const uint8_t model_bin_end[] asm("_binary_model_bin_end");

/* ---- 画面レイアウト（1280×720 横向き） ---------------------------------- */
#define PF_SCALE 5
#define PF_IMG_PX (RF_IMG_HW * PF_SCALE) /* 640 */
#define PF_IMG_X 40
#define PF_IMG_Y 40
#define PF_PANEL_X 720
#define PF_PANEL_W 520

/* 5 クラス = 性別 × 笑顔 + null（data/make_ffhq_labels.py）。クラス n_cond-1 は無条件で、
 * CFG の負例にも使われる（このクラスを選ぶと guidance は no-op） */
static const char *k_class_names[] = {"female / no-smile", "female / smile", "male / no-smile",
                                      "male / smile", "null (unconditional)"};

typedef struct {
    uint64_t seed;
    int k_steps;
    int cond;
    int w_idx;
    bool from_serial;
} pf_req_t;

static rf_model_t *s_model;          /* PSRAM（~30 KB） */
static uint8_t *s_img;               /* PSRAM: 128×128×3 */
static lgfx::rgb888_t *s_scaled;     /* PSRAM: 640×640×3 */
static QueueHandle_t s_q;
static SemaphoreHandle_t s_gfx;      /* M5GFX はスレッド非対応。描画はこのロック下で */
static pf_req_t s_cur;               /* 最後に表示した / 進行中のリクエスト */
static int64_t s_t0_us;
static uint32_t s_last_ms;
/* 起動直後にタッチパネルが click を報告し、勝手に 1 枚生成した（実機で 2 回連続）。
 * 起動から 3 秒間はタッチを無視する */
static int64_t s_touch_ok_us;

static int w_value(int w_idx) {
    if (w_idx < 0 || w_idx >= (int)s_model->n_w) return 0;
    return (int)(s_model->w_q8[w_idx] / 256);
}

/* ---- 描画 --------------------------------------------------------------- */
static void draw_panel(const char *status, uint16_t color) {
    auto &d = M5.Display;
    d.startWrite();
    d.fillRect(PF_PANEL_X, 0, PF_PANEL_W, d.height(), TFT_BLACK);
    d.setTextDatum(top_left);
    d.setFont(&fonts::FreeSansBold18pt7b);
    d.setTextColor(TFT_WHITE, TFT_BLACK);
    d.drawString("pico-faces / Tab5", PF_PANEL_X, 40);
    d.setFont(&fonts::FreeSans12pt7b);
    d.setTextColor(TFT_LIGHTGREY, TFT_BLACK);
    d.drawString("DiT + VAE, int8, ESP32-P4 x2", PF_PANEL_X, 84);
    char buf[96];
    int y = 150;
    d.setFont(&fonts::FreeSans18pt7b);
    d.setTextColor(TFT_WHITE, TFT_BLACK);
    snprintf(buf, sizeof buf, "seed  %llu", (unsigned long long)s_cur.seed);
    d.drawString(buf, PF_PANEL_X, y); y += 44;
    snprintf(buf, sizeof buf, "class %d  %s", s_cur.cond,
             s_cur.cond < 5 ? k_class_names[s_cur.cond] : "");
    d.drawString(buf, PF_PANEL_X, y); y += 44;
    if (s_cur.w_idx < 0) snprintf(buf, sizeof buf, "steps %d   cfg none", s_cur.k_steps);
    else snprintf(buf, sizeof buf, "steps %d   cfg w=%d", s_cur.k_steps, w_value(s_cur.w_idx));
    d.drawString(buf, PF_PANEL_X, y); y += 44;
    if (s_last_ms) {
        snprintf(buf, sizeof buf, "time  %.2f s", s_last_ms / 1000.0);
        d.drawString(buf, PF_PANEL_X, y);
    }
    y += 70;
    d.setFont(&fonts::FreeSansBold12pt7b);
    d.setTextColor(color, TFT_BLACK);
    d.drawString(status, PF_PANEL_X, y);
    /* 操作ヒント */
    d.setFont(&fonts::FreeSans9pt7b);
    d.setTextColor(TFT_DARKGREY, TFT_BLACK);
    int hy = d.height() - 120;
    d.drawString("tap image : next seed", PF_PANEL_X, hy);
    d.drawString("tap here (upper) : next class", PF_PANEL_X, hy + 26);
    d.drawString("tap here (lower) : cfg none / 4 / 6 / 8", PF_PANEL_X, hy + 52);
    d.drawString("serial: G <seed> [k] [class] [w]", PF_PANEL_X, hy + 78);
    d.endWrite();
}

static void draw_progress(void) {
    auto &d = M5.Display;
    const int x = PF_PANEL_X, y = 380, w = PF_PANEL_W - 40, h = 18;
    int total = rf_progress_total ? rf_progress_total : 1;
    int done = rf_progress > total ? total : rf_progress;
    d.startWrite();
    d.drawRect(x, y, w, h, TFT_DARKGREY);
    d.fillRect(x + 2, y + 2, (w - 4) * done / total, h - 4, TFT_GREENYELLOW);
    d.fillRect(x + 2 + (w - 4) * done / total, y + 2, (w - 4) - (w - 4) * done / total, h - 4, TFT_BLACK);
    char buf[48];
    if (rf_progress == 0) snprintf(buf, sizeof buf, "done");
    else if (rf_progress > (uint8_t)(total - 1)) snprintf(buf, sizeof buf, "VAE decode");
    else snprintf(buf, sizeof buf, "DiT step %d / %d", (int)rf_progress, total - 1);
    d.setFont(&fonts::FreeSans9pt7b);
    d.setTextDatum(top_left);
    d.setTextColor(TFT_LIGHTGREY, TFT_BLACK);
    d.fillRect(x, y + h + 6, w, 22, TFT_BLACK);
    d.drawString(buf, x, y + h + 6);
    d.endWrite();
}

/* 128×128 RGB (HWC) → 640×640 に最近傍で拡大して表示 */
static void draw_image(void) {
    for (int y = 0; y < PF_IMG_PX; y++) {
        const uint8_t *src = s_img + (size_t)(y / PF_SCALE) * RF_IMG_HW * RF_IMG_CH;
        lgfx::rgb888_t *dst = s_scaled + (size_t)y * PF_IMG_PX;
        for (int x = 0; x < PF_IMG_PX; x++) {
            const uint8_t *p = src + (size_t)(x / PF_SCALE) * RF_IMG_CH;
            dst[x].r = p[0];
            dst[x].g = p[1];
            dst[x].b = p[2];
        }
    }
    auto &d = M5.Display;
    d.startWrite();
    d.pushImage(PF_IMG_X, PF_IMG_Y, PF_IMG_PX, PF_IMG_PX, s_scaled);
    d.endWrite();
}

/* dit.c から各ステップの境目で呼ばれる（生成タスク上。ワーカーとは同時に走らない） */
extern "C" void rf_step_hook(void) {
    if (xSemaphoreTake(s_gfx, pdMS_TO_TICKS(50)) == pdTRUE) {
        draw_progress();
        xSemaphoreGive(s_gfx);
    }
}

/* ---- 生成タスク（コア 0 固定） ---------------------------------------- */
static void gen_task(void *arg) {
    (void)arg;
    pf_req_t req;
    for (;;) {
        if (xQueueReceive(s_q, &req, pdMS_TO_TICKS(20)) != pdTRUE) {
            /* 待機中: タッチを見る */
            M5.update();
            auto n = M5.Touch.getCount();
            if (esp_timer_get_time() < s_touch_ok_us) continue; /* 起動直後の誤検知を捨てる */
            bool fire = false;
            pf_req_t nx = s_cur;
            nx.from_serial = false;
            for (size_t i = 0; i < n && !fire; ++i) {
                auto t = M5.Touch.getDetail(i);
                if (!t.wasClicked()) continue;
                /* 起動後しばらく、触っていないのに click が来ることがある（実機で再現）。座標を残す */
                ESP_LOGI(TAG, "touch click at %d,%d", (int)t.x, (int)t.y);
                if (t.x >= PF_IMG_X && t.x < PF_IMG_X + PF_IMG_PX && t.y >= PF_IMG_Y && t.y < PF_IMG_Y + PF_IMG_PX) {
                    nx.seed = s_cur.seed + 1;
                    fire = true;
                } else if (t.x >= PF_PANEL_X) {
                    if (t.y < M5.Display.height() / 2) {
                        nx.cond = (s_cur.cond + 1) % (int)s_model->n_cond;
                    } else {
                        /* none → w[0] → w[1] → … → none */
                        nx.w_idx = s_cur.w_idx + 1;
                        if (nx.w_idx >= (int)s_model->n_w) nx.w_idx = -1;
                    }
                    fire = true;
                }
            }
            if (!fire) continue;
            req = nx;
        }
        s_cur = req;
        s_last_ms = 0;
        xSemaphoreTake(s_gfx, portMAX_DELAY);
        draw_panel("generating ...", TFT_YELLOW);
        rf_progress = 0;
        draw_progress();
        xSemaphoreGive(s_gfx);

        pf_par_reset();
        s_t0_us = esp_timer_get_time();
        rf_generate(s_model, req.seed, req.k_steps, req.cond, req.w_idx, s_img, NULL);
        s_last_ms = (uint32_t)((esp_timer_get_time() - s_t0_us) / 1000);
        uint32_t crc = rf_crc32(s_img, (size_t)RF_IMG_HW * RF_IMG_HW * RF_IMG_CH);

        xSemaphoreTake(s_gfx, portMAX_DELAY);
        draw_image();
        draw_panel("done", TFT_GREENYELLOW);
        draw_progress();
        xSemaphoreGive(s_gfx);

        printf("OK seed=%llu k=%d class=%d w=%d crc32=%08" PRIx32 " ms=%" PRIu32 "\n",
               (unsigned long long)req.seed, req.k_steps, req.cond, w_value(req.w_idx), crc, s_last_ms);
        ESP_LOGI(TAG, "seed %llu class %d k %d w %d: crc32 %08" PRIx32 " in %" PRIu32 " ms",
                 (unsigned long long)req.seed, req.cond, req.k_steps, w_value(req.w_idx), crc, s_last_ms);
#ifdef RF_PIE_P4
        ESP_LOGI(TAG, "  DiT %lld ms, VAE decode %lld ms (%s)", (long long)(s_last_ms - pf_vae_us / 1000),
                 (long long)(pf_vae_us / 1000), pf_vae_pie ? "PIE" : "reference");
#endif
        pf_par_report();
    }
}

/* ---- USB シリアルの行入力 ---------------------------------------------- */
static void console_task(void *arg) {
    (void)arg;
    usb_serial_jtag_driver_config_t cfg = USB_SERIAL_JTAG_DRIVER_CONFIG_DEFAULT();
    cfg.rx_buffer_size = 1024;
    cfg.tx_buffer_size = 1024;
    esp_err_t err = usb_serial_jtag_driver_install(&cfg);
    if (err != ESP_OK && err != ESP_ERR_INVALID_STATE) {
        ESP_LOGE(TAG, "usb_serial_jtag_driver_install: %s", esp_err_to_name(err));
        vTaskDelete(NULL);
        return;
    }
    char line[96];
    int n = 0;
    for (;;) {
        uint8_t c;
        if (usb_serial_jtag_read_bytes(&c, 1, pdMS_TO_TICKS(100)) != 1) continue;
        if (c != '\n' && c != '\r') {
            if (n < (int)sizeof line - 1) line[n++] = (char)c;
            continue;
        }
        line[n] = 0;
        n = 0;
        if (line[0] == 'G') {
            char *e1, *e2, *e3, *e4;
            pf_req_t r;
            r.from_serial = true;
            r.seed = strtoull(line + 1, &e1, 0);
            r.k_steps = (int)strtol(e1, &e2, 0);
            if (!r.k_steps) r.k_steps = 4;
            long cv = strtol(e2, &e3, 0);
            r.cond = (e3 != e2) ? (int)cv : (int)(r.seed % s_model->n_cond);
            long wv = strtol(e3, &e4, 0);
            r.w_idx = -1;
            if (e4 != e3) {
                for (uint32_t j = 0; j < s_model->n_w; j++)
                    if (s_model->w_q8[j] == (uint32_t)(wv * 256)) r.w_idx = (int)j;
            } else if (e3 == e2 && s_model->n_w) {
                r.w_idx = (int)(r.seed % (s_model->n_w + 1)) - 1;
            }
            if (xQueueSend(s_q, &r, 0) != pdTRUE) printf("BUSY\n");
        } else if (line[0] == 'I') {
            printf("pico-faces-tab5 K=%u dim=%u depth=%u cond=%u ch=%u n_w=%u blob=%u sys=%dMHz\n",
                   (unsigned)s_model->K, (unsigned)s_model->dim, (unsigned)s_model->depth,
                   (unsigned)s_model->n_cond, (unsigned)s_model->img_ch, (unsigned)s_model->n_w,
                   (unsigned)(model_bin_end - model_bin_start), CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ);
        } else if (line[0]) {
            printf("? (G <seed> [k] [class] [w] | I)\n");
        }
    }
}

extern "C" void app_main(void) {
    auto cfg = M5.config();
    cfg.clear_display = true;
    cfg.internal_mic = false;
    cfg.internal_spk = false;
    M5.begin(cfg);
    auto &d = M5.Display;
    if (d.height() > d.width()) d.setRotation((d.getRotation() + 3) & 3); /* Tab5 は縦長で起動する */
    d.fillScreen(TFT_BLACK);
    ESP_LOGI(TAG, "board %d, display %d x %d", (int)M5.getBoard(), (int)d.width(), (int)d.height());

    /* モデル: flash の埋め込み blob → PSRAM（エンジンは 4 バイト境界の配列をそのまま指す） */
    size_t blob_len = (size_t)(model_bin_end - model_bin_start);
    uint8_t *blob = (uint8_t *)heap_caps_aligned_alloc(16, blob_len, MALLOC_CAP_SPIRAM);
    s_model = (rf_model_t *)heap_caps_calloc(1, sizeof(rf_model_t), MALLOC_CAP_SPIRAM);
    s_img = (uint8_t *)heap_caps_malloc((size_t)RF_IMG_HW * RF_IMG_HW * RF_IMG_CH, MALLOC_CAP_SPIRAM);
    s_scaled = (lgfx::rgb888_t *)heap_caps_malloc((size_t)PF_IMG_PX * PF_IMG_PX * sizeof(lgfx::rgb888_t), MALLOC_CAP_SPIRAM);
    if (!blob || !s_model || !s_img || !s_scaled) {
        ESP_LOGE(TAG, "PSRAM 確保に失敗（blob %u B）", (unsigned)blob_len);
        return;
    }
    memcpy(blob, model_bin_start, blob_len);
    int rc = rf_model_load(blob, blob_len, s_model);
    if (rc != 0) {
        ESP_LOGE(TAG, "rf_model_load: %d（blob と rf_cfg.h のモデルが食い違っていないか）", rc);
        d.setFont(&fonts::FreeSansBold18pt7b);
        d.drawString("model load failed", 40, 40);
        return;
    }
    ESP_LOGI(TAG, "model: K=%u dim=%u depth=%u cond=%u n_w=%u blob=%u B; internal free %u B, largest %u B",
             (unsigned)s_model->K, (unsigned)s_model->dim, (unsigned)s_model->depth, (unsigned)s_model->n_cond,
             (unsigned)s_model->n_w, (unsigned)blob_len,
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
             (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL));

    s_gfx = xSemaphoreCreateMutex();
    s_q = xQueueCreate(4, sizeof(pf_req_t));
    s_touch_ok_us = esp_timer_get_time() + 3000000;
#ifdef RF_PIE_P4
    {
        int64_t t = esp_timer_get_time();
        int f = pf_pie_selftest();
        ESP_LOGI(TAG, "PIE selftest: %s (%lld us)", f ? "FAILED -> scalar" : "OK", (long long)(esp_timer_get_time() - t));
        pf_pie_bench();
        pf_pie_bench3();
        t = esp_timer_get_time();
        int nT = pf_stage_prepare(s_model);
        ESP_LOGI(TAG, "PIE: %d transposed weight copies in PSRAM (%lld us)", nT, (long long)(esp_timer_get_time() - t));
    }
#endif
    pf_par_init();

    /* 最初の 1 枚: README の例（seed 3, K=8, class 4 → こちらでは class 3, w=6）に近い設定 */
    pf_req_t first;
    first.seed = 3;
    first.k_steps = (int)s_model->K;   /* 8 */
    first.cond = 3;
    first.w_idx = -1;
    for (uint32_t j = 0; j < s_model->n_w; j++)
        if (s_model->w_q8[j] == 6 * 256) first.w_idx = (int)j;
    first.from_serial = false;
    s_cur = first;
    xQueueSend(s_q, &first, 0);

    xTaskCreatePinnedToCore(gen_task, "pf_gen", 20480, NULL, 5, NULL, 0);
    xTaskCreatePinnedToCore(console_task, "pf_con", 6144, NULL, 4, NULL, 1);
}
