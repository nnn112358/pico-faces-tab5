/* pico-faces on M5Stack Tab5（ESP32-P4）
 *
 * 上流 cpldcpu/pico-faces の C99 推論エンジン（upstream/engine）で 128×128 RGB の顔画像を
 * 生成し、Tab5 の 1280×720 画面に 5 倍で表示する。RP2350 版ファーム（firmware/main.c）の
 * 置き換えで、エンジンのソースは 1 バイトも変えていない。
 *
 * 構成:
 *   - モデル blob は app（.rodata）に埋め込み、起動時に PSRAM へコピーして読む
 *   - 2 コア並列は pf_par.c（rf_par_for の置き換え）。生成タスクはコア 0 固定
 *   - 操作は右パネルのボタン（seed の増減 / class / cfg / 1 枚生成 / 10 枚連続）。
 *     USB シリアルからは上流と同じ `G <seed> [k] [class] [w]` も受ける
 *
 * タッチ判定:
 *   M5Unified の wasClicked() は「押した位置から 8 px も動かず 500 ms 以内に離した」ときしか真に
 *   ならず、指で普通にタップすると外れる（実機で 40 秒タップして 0 回だった）。
 *   wasReleased()（離した瞬間）と base_x/base_y（押し始めの座標）で判定する。
 *   生成中も rf_par_for 経由（pf_ui_poll、40 ms 間隔）で読み、触れていれば印を立てる。
 *   1 枚生成ではその印を捨て、10 枚連続では次の画像に進まず「中断」にする。
 *
 * USB シリアル（115200 / USB Serial-JTAG）:
 *   G <seed> [k_steps] [class] [w] [count]   生成。class/w を省略すると上流の golden 規約
 *                                            （class = seed % n_cond, w = seed % (n_w+1) - 1）。
 *                                            count（既定 1）枚を seed から順に生成
 *   I                                        モデル情報
 *   応答: OK seed=.. k=.. class=.. w=.. crc32=........ ms=..（1 枚ごと）
 *   → ホストの rf_golden と同じ crc32 が出れば bit 一致（docs/details.md 参照）
 */
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <M5Unified.h>

#include "driver/usb_serial_jtag.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
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
    int count;       /* seed から順に何枚生成するか */
    bool from_serial;
} pf_req_t;

static rf_model_t *s_model;          /* PSRAM（~30 KB） */
static uint8_t *s_img;               /* PSRAM: 128×128×3 */
static lgfx::rgb888_t *s_scaled;     /* PSRAM: 640×640×3 */
static QueueHandle_t s_q;
static SemaphoreHandle_t s_gfx;      /* M5GFX はスレッド非対応。描画はこのロック下で */

/* パネルの設定値（次に生成するもの） */
static uint64_t s_seed = 3;
static int s_cond = 3;
static int s_w_idx = -1;
static int s_k = 8;

/* 最後に生成した / 生成中のもの */
static pf_req_t s_cur;
static uint32_t s_last_ms;
static uint32_t s_last_crc;
static int s_batch_i, s_batch_n;     /* 連続生成の進捗（1 始まり / 全体） */
static bool s_busy;

/* 起動直後にタッチパネルが離しイベントを報告することがある。起動から 3 秒間は無視する */
static int64_t s_touch_ok_us;
/* 生成中に画面が触られたか（pf_ui_poll が立て、連続生成の中断に使う） */
static volatile bool s_touched_while_busy;

static int w_value(int w_idx) {
    if (w_idx < 0 || w_idx >= (int)s_model->n_w) return 0;
    return (int)(s_model->w_q8[w_idx] / 256);
}

/* ---- ボタン ------------------------------------------------------------- */
enum {
    B_SEED_M10, B_SEED_M1, B_SEED_P1, B_SEED_P10,
    B_CLASS, B_CFG,
    B_GEN1, B_GEN10,
    B_N,
    B_IMAGE = 100,   /* 画像をタップ = 1 枚生成 */
    B_OTHER = 101    /* ボタン以外 */
};

typedef struct {
    int x, y, w, h;
} pf_rect_t;

/* seed の行: [-10][-1]  seed  [+1][+10] */
#define ROW_SEED_Y 150
#define ROW_CLASS_Y 240
#define ROW_CFG_Y 310
#define ROW_GEN_Y 400
#define ROW_PROG_Y 520
#define ROW_STATUS_Y 580
static const pf_rect_t k_btn[B_N] = {
    {PF_PANEL_X, ROW_SEED_Y, 92, 60},            /* -10 */
    {PF_PANEL_X + 100, ROW_SEED_Y, 92, 60},      /* -1 */
    {PF_PANEL_X + 328, ROW_SEED_Y, 92, 60},      /* +1 */
    {PF_PANEL_X + 428, ROW_SEED_Y, 92, 60},      /* +10 */
    {PF_PANEL_X, ROW_CLASS_Y, PF_PANEL_W, 56},   /* class */
    {PF_PANEL_X, ROW_CFG_Y, PF_PANEL_W, 56},     /* cfg */
    {PF_PANEL_X, ROW_GEN_Y, 252, 96},            /* 1 枚生成 */
    {PF_PANEL_X + 268, ROW_GEN_Y, 252, 96},      /* 10 枚連続 */
};

static bool in_rect(const pf_rect_t &r, int x, int y) {
    return x >= r.x && x < r.x + r.w && y >= r.y && y < r.y + r.h;
}

static void draw_button(const pf_rect_t &r, const char *label, uint16_t fill, uint16_t fg) {
    auto &d = M5.Display;
    d.fillRoundRect(r.x, r.y, r.w, r.h, 10, fill);
    d.drawRoundRect(r.x, r.y, r.w, r.h, 10, TFT_DARKGREY);
    d.setTextDatum(middle_center);
    d.setTextColor(fg, fill);
    d.drawString(label, r.x + r.w / 2, r.y + r.h / 2);
}

/* ---- 描画 --------------------------------------------------------------- */
static void draw_status(const char *status, uint16_t color) {
    auto &d = M5.Display;
    d.startWrite();
    d.fillRect(PF_PANEL_X, ROW_STATUS_Y, PF_PANEL_W, 40, TFT_BLACK);
    d.setFont(&fonts::lgfxJapanGothic_24);
    d.setTextDatum(top_left);
    d.setTextColor(color, TFT_BLACK);
    d.drawString(status, PF_PANEL_X, ROW_STATUS_Y);
    d.endWrite();
}

static void draw_panel(void) {
    auto &d = M5.Display;
    char buf[96];
    d.startWrite();
    d.fillRect(PF_PANEL_X, 0, PF_PANEL_W, ROW_STATUS_Y, TFT_BLACK);
    d.setTextDatum(top_left);
    d.setFont(&fonts::FreeSansBold18pt7b);
    d.setTextColor(TFT_WHITE, TFT_BLACK);
    d.drawString("pico-faces / Tab5", PF_PANEL_X, 30);
    d.setFont(&fonts::FreeSans12pt7b);
    d.setTextColor(TFT_LIGHTGREY, TFT_BLACK);
    d.drawString("DiT + VAE, int8, ESP32-P4 PIE x2", PF_PANEL_X, 74);

    /* seed の行 */
    d.setFont(&fonts::lgfxJapanGothic_20);
    d.setTextColor(TFT_LIGHTGREY, TFT_BLACK);
    d.drawString("seed（次に生成する番号）", PF_PANEL_X, ROW_SEED_Y - 28);
    d.setFont(&fonts::lgfxJapanGothic_24);
    draw_button(k_btn[B_SEED_M10], "-10", TFT_DARKGREY, TFT_WHITE);
    draw_button(k_btn[B_SEED_M1], "-1", TFT_DARKGREY, TFT_WHITE);
    draw_button(k_btn[B_SEED_P1], "+1", TFT_DARKGREY, TFT_WHITE);
    draw_button(k_btn[B_SEED_P10], "+10", TFT_DARKGREY, TFT_WHITE);
    d.setFont(&fonts::lgfxJapanGothic_28);
    d.setTextDatum(middle_center);
    d.setTextColor(TFT_WHITE, TFT_BLACK);
    snprintf(buf, sizeof buf, "%llu", (unsigned long long)s_seed);
    d.drawString(buf, PF_PANEL_X + 260, ROW_SEED_Y + 30);

    /* class / cfg */
    d.setFont(&fonts::lgfxJapanGothic_24);
    snprintf(buf, sizeof buf, "class %d : %s", s_cond, s_cond < 5 ? k_class_names[s_cond] : "");
    draw_button(k_btn[B_CLASS], buf, TFT_DARKGREY, TFT_WHITE);
    if (s_w_idx < 0) snprintf(buf, sizeof buf, "cfg : none   (steps %d)", s_k);
    else snprintf(buf, sizeof buf, "cfg : w=%d   (steps %d)", w_value(s_w_idx), s_k);
    draw_button(k_btn[B_CFG], buf, TFT_DARKGREY, TFT_WHITE);

    /* 生成ボタン */
    d.setFont(&fonts::lgfxJapanGothic_28);
    draw_button(k_btn[B_GEN1], "1枚生成", 0x0320 /* 濃い緑 */, TFT_WHITE);
    draw_button(k_btn[B_GEN10], "10枚連続", 0x0014 /* 濃い青 */, TFT_WHITE);

    /* 最後の結果 */
    d.setFont(&fonts::lgfxJapanGothic_20);
    d.setTextDatum(top_left);
    d.setTextColor(TFT_LIGHTGREY, TFT_BLACK);
    if (s_last_ms) {
        snprintf(buf, sizeof buf, "表示中: seed %llu  class %d  %s  %.2f s", (unsigned long long)s_cur.seed,
                 s_cur.cond, s_cur.w_idx < 0 ? "cfg none" : "cfg on", s_last_ms / 1000.0);
        d.drawString(buf, PF_PANEL_X, ROW_STATUS_Y + 44);
        snprintf(buf, sizeof buf, "crc32 %08" PRIx32 "   シリアル: G <seed> [k] [class] [w]", s_last_crc);
        d.drawString(buf, PF_PANEL_X, ROW_STATUS_Y + 70);
    } else {
        d.fillRect(PF_PANEL_X, ROW_STATUS_Y + 44, PF_PANEL_W, 60, TFT_BLACK);
    }
    d.endWrite();
}

static void draw_progress(void) {
    auto &d = M5.Display;
    const int x = PF_PANEL_X, y = ROW_PROG_Y, w = PF_PANEL_W, h = 18;
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

/* ---- タッチ ------------------------------------------------------------- */
/* M5.update() を 1 回呼び、離しイベントがあれば押し始めの座標で当たり判定する。
 * 戻り値: ボタン番号 / B_IMAGE / B_OTHER、何も無ければ -1 */
static int poll_touch(void) {
    M5.update();
    if (esp_timer_get_time() < s_touch_ok_us) return -1;
    int n = M5.Touch.getCount();
    for (int i = 0; i < n; i++) {
        auto t = M5.Touch.getDetail(i);
        if (!t.wasReleased()) continue;
        int x = t.base_x, y = t.base_y;
        ESP_LOGI(TAG, "touch release: began %d,%d ended %d,%d", x, y, (int)t.x, (int)t.y);
        if (x >= PF_IMG_X && x < PF_IMG_X + PF_IMG_PX && y >= PF_IMG_Y && y < PF_IMG_Y + PF_IMG_PX) return B_IMAGE;
        for (int b = 0; b < B_N; b++)
            if (in_rect(k_btn[b], x, y)) return b;
        return B_OTHER;
    }
    return -1;
}

/* 生成中のタッチ監視。pf_par.c の rf_par_for から呼ばれる（生成タスク上、40 ms に 1 回に間引く）。
 * M5Unified は「押し始めを見ていない離し」を報告しないので、画像と画像の間で 1 回読むだけでは
 * 生成中に完結したタップが見えない。生成の内側で読んで、触れていたら印を立てる */
extern "C" void pf_ui_poll(void) {
    static int64_t last_us;
    if (!s_busy) return;
    int64_t now = esp_timer_get_time();
    if (now - last_us < 40000) return;
    last_us = now;
    M5.update();
    if (now < s_touch_ok_us) return;
    int n = M5.Touch.getCount();
    for (int i = 0; i < n; i++) {
        auto t = M5.Touch.getDetail(i);
        if (t.isPressed() || t.wasReleased()) {
            if (!s_touched_while_busy) ESP_LOGI(TAG, "touch during generation at %d,%d", (int)t.x, (int)t.y);
            s_touched_while_busy = true;
        }
    }
}

/* 生成中に溜まったタッチを読み捨てる（生成が終わった直後に 1 回） */
static void drain_touch(void) {
    for (int i = 0; i < 3; i++) {
        M5.update();
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}

/* ---- 生成 --------------------------------------------------------------- */
static void generate_one(const pf_req_t &r) {
    s_cur = r;
    s_busy = true;
    char buf[80];
    xSemaphoreTake(s_gfx, portMAX_DELAY);
    if (s_batch_n > 1) snprintf(buf, sizeof buf, "生成中 %d / %d  seed %llu（画面を触ると中断）", s_batch_i, s_batch_n,
                                (unsigned long long)r.seed);
    else snprintf(buf, sizeof buf, "生成中  seed %llu", (unsigned long long)r.seed);
    draw_status(buf, TFT_YELLOW);
    rf_progress = 0;
    draw_progress();
    xSemaphoreGive(s_gfx);

    pf_par_reset();
    int64_t t0 = esp_timer_get_time();
    rf_generate(s_model, r.seed, r.k_steps, r.cond, r.w_idx, s_img, NULL);
    s_last_ms = (uint32_t)((esp_timer_get_time() - t0) / 1000);
    s_last_crc = rf_crc32(s_img, (size_t)RF_IMG_HW * RF_IMG_HW * RF_IMG_CH);

    xSemaphoreTake(s_gfx, portMAX_DELAY);
    draw_image();
    draw_panel();
    draw_progress();
    xSemaphoreGive(s_gfx);

    printf("OK seed=%llu k=%d class=%d w=%d crc32=%08" PRIx32 " ms=%" PRIu32 "\n", (unsigned long long)r.seed,
           r.k_steps, r.cond, w_value(r.w_idx), s_last_crc, s_last_ms);
    ESP_LOGI(TAG, "seed %llu class %d k %d w %d: crc32 %08" PRIx32 " in %" PRIu32 " ms", (unsigned long long)r.seed,
             r.cond, r.k_steps, w_value(r.w_idx), s_last_crc, s_last_ms);
#ifdef RF_PIE_P4
    ESP_LOGI(TAG, "  DiT %lld ms, VAE decode %lld ms (%s)", (long long)(s_last_ms - pf_vae_us / 1000),
             (long long)(pf_vae_us / 1000), pf_vae_pie ? "PIE" : "reference");
#endif
    pf_par_report();
    s_busy = false;
}

/* パネルの設定から要求を作り、キューに積む。seed は count ぶん進める */
static void enqueue_from_panel(int count) {
    pf_req_t r;
    r.seed = s_seed;
    r.k_steps = s_k;
    r.cond = s_cond;
    r.w_idx = s_w_idx;
    r.count = count;
    r.from_serial = false;
    if (xQueueSend(s_q, &r, 0) == pdTRUE) s_seed += (uint64_t)count;
}

/* ---- 生成タスク（コア 0 固定） ---------------------------------------- */
static void gen_task(void *arg) {
    (void)arg;
    pf_req_t req;
    for (;;) {
        if (xQueueReceive(s_q, &req, pdMS_TO_TICKS(20)) != pdTRUE) {
            /* 待機中: タッチを見る */
            int b = poll_touch();
            if (b < 0 || b == B_OTHER) continue;
            switch (b) {
            case B_SEED_M10: s_seed = s_seed >= 10 ? s_seed - 10 : 0; break;
            case B_SEED_M1: s_seed = s_seed >= 1 ? s_seed - 1 : 0; break;
            case B_SEED_P1: s_seed += 1; break;
            case B_SEED_P10: s_seed += 10; break;
            case B_CLASS: s_cond = (s_cond + 1) % (int)s_model->n_cond; break;
            case B_CFG:
                s_w_idx += 1;
                if (s_w_idx >= (int)s_model->n_w) s_w_idx = -1;
                break;
            case B_GEN1:
            case B_IMAGE: enqueue_from_panel(1); break;
            case B_GEN10: enqueue_from_panel(10); break;
            default: break;
            }
            xSemaphoreTake(s_gfx, portMAX_DELAY);
            draw_panel();
            xSemaphoreGive(s_gfx);
            continue;
        }
        /* 1 枚、または seed から count 枚を順に */
        int n = req.count > 0 ? req.count : 1;
        s_batch_n = n;
        bool cancelled = false;
        for (int i = 0; i < n; i++) {
            s_batch_i = i + 1;
            pf_req_t r = req;
            r.seed = req.seed + (uint64_t)i;
            s_touched_while_busy = false;
            generate_one(r);
            if (i + 1 < n) {
                /* 生成中に画面を触っていたら（pf_ui_poll が見ている）ここで止める */
                int b = poll_touch();
                if (s_touched_while_busy || b >= 0) {
                    cancelled = true;
                    break;
                }
            }
        }
        s_touched_while_busy = false;
        drain_touch();
        char buf[80];
        if (cancelled) snprintf(buf, sizeof buf, "中断しました（%d / %d 枚）", s_batch_i, n);
        else if (n > 1) snprintf(buf, sizeof buf, "完了  %d 枚（seed %llu 〜 %llu）", n, (unsigned long long)req.seed,
                                 (unsigned long long)(req.seed + n - 1));
        else snprintf(buf, sizeof buf, "完了  seed %llu  %.2f s", (unsigned long long)req.seed, s_last_ms / 1000.0);
        s_batch_n = 0;
        xSemaphoreTake(s_gfx, portMAX_DELAY);
        draw_status(buf, TFT_GREENYELLOW);
        xSemaphoreGive(s_gfx);
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
            char *e1, *e2, *e3, *e4, *e5;
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
            long cnt = strtol(e4, &e5, 0);
            r.count = (e5 != e4 && cnt > 0) ? (int)cnt : 1;
            if (xQueueSend(s_q, &r, 0) != pdTRUE) printf("BUSY\n");
        } else if (line[0] == 'I') {
            printf("pico-faces-tab5 K=%u dim=%u depth=%u cond=%u ch=%u n_w=%u blob=%u sys=%dMHz\n",
                   (unsigned)s_model->K, (unsigned)s_model->dim, (unsigned)s_model->depth,
                   (unsigned)s_model->n_cond, (unsigned)s_model->img_ch, (unsigned)s_model->n_w,
                   (unsigned)(model_bin_end - model_bin_start), CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ);
        } else if (line[0]) {
            printf("? (G <seed> [k] [class] [w] [count] | I)\n");
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

    /* パネルの初期値: seed 3、K ステップ、class 3（male / smile）、cfg none。最初の 1 枚をすぐ生成する。
     * cfg none は 1 ステップにつき DiT を 1 回しか通さないので、w=6 の約半分の時間で済む
     * （K=8 で約 2.0 秒。w=6 だと約 3.5 秒）。cfg ボタンで none → 4 → 6 → 8 と切り替えられる */
    s_k = (int)s_model->K; /* 8 */
    s_cond = 3;
    s_w_idx = -1;          /* cfg none（高速） */
    s_seed = 3;
    s_cur.seed = s_seed;
    s_cur.k_steps = s_k;
    s_cur.cond = s_cond;
    s_cur.w_idx = s_w_idx;
    draw_panel();
    enqueue_from_panel(1);

    xTaskCreatePinnedToCore(gen_task, "pf_gen", 20480, NULL, 5, NULL, 0);
    xTaskCreatePinnedToCore(console_task, "pf_con", 6144, NULL, 4, NULL, 1);
}
