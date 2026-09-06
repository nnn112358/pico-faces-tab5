/* 2 コア並列: エンジンの weak な rf_par_for / rf_core_id を FreeRTOS で置き換える。
 *
 * 上流の firmware/par.c（RP2350: core1 を FIFO で叩く）と同じ契約:
 *   呼び出し側（コア 0）が [0, n/2) を、ワーカー（コア 1）が [n/2, n) を処理する。
 *   書き込みは互いに素なのでロック不要。通知（task notification）で go / done を渡す。
 *
 * ⚠️ rf_core_id() は VAE デコーダの行圧縮スクラッチ rowbuf[2] の添字に使われる。
 *    生成タスクはコア 0 に、ワーカーはコア 1 に**固定**しないと 2 つのタスクが同じ
 *    スクラッチを触る。
 *
 * プロファイル: rf_par_for の呼び出しを関数ポインタごとに集計し、重みステージング
 * （weak な rf_stage_start の memcpy）の時間も足す。pf_par_report() で表示。
 * 関数の実体は `nm build/pico_faces_tab5.elf | grep _range` でアドレスから引く。 */
#include <stdio.h>
#include <string.h>

#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "rf_ops.h"

typedef void (*par_fn)(int, int, void *);
static TaskHandle_t s_worker;
static TaskHandle_t s_caller;
static par_fn volatile g_fn;
static void *volatile g_ctx;
static int volatile g_i0, g_i1;

#define PF_PROF_N 24
typedef struct {
    par_fn fn;
    int64_t us;
    uint32_t calls;
} pf_prof_t;
static pf_prof_t s_prof[PF_PROF_N];
static int s_nprof;
int64_t pf_stage_us;       /* rf_stage_start の集計（PIE 版は pf_pie_linear.c が加算する） */
uint32_t pf_stage_calls;
size_t pf_stage_bytes;

static void worker_task(void *arg) {
    (void)arg;
    for (;;) {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY); /* go */
        g_fn(g_i0, g_i1, g_ctx);
        __sync_synchronize();
        xTaskNotifyGive(s_caller); /* done */
    }
}

void pf_par_init(void) {
    /* 生成タスクより高い優先度: 半分の仕事を任せたら即座に走り出してほしい。
     * スタックは softmax の e[512] や MLP の h1[1024] など数 KB の局所配列を見込む。 */
    xTaskCreatePinnedToCore(worker_task, "pf_par1", 16384, NULL,
                            configMAX_PRIORITIES - 2, &s_worker, 1);
}

int rf_core_id(void) { return (int)xPortGetCoreID(); }

static void prof_add(par_fn fn, int64_t us) {
    for (int i = 0; i < s_nprof; i++)
        if (s_prof[i].fn == fn) {
            s_prof[i].us += us;
            s_prof[i].calls++;
            return;
        }
    if (s_nprof < PF_PROF_N) {
        s_prof[s_nprof].fn = fn;
        s_prof[s_nprof].us = us;
        s_prof[s_nprof].calls = 1;
        s_nprof++;
    }
}

void pf_ui_poll(void); /* main.cpp: 生成中のタッチ監視（内部で 40 ms に間引く） */

void rf_par_for(int n, void (*fn)(int, int, void *), void *ctx) {
    pf_ui_poll();
    int64_t t0 = esp_timer_get_time();
    int mid = n / 2;
    if (mid == 0 || s_worker == NULL) {
        fn(0, n, ctx);
        prof_add(fn, esp_timer_get_time() - t0);
        return;
    }
    s_caller = xTaskGetCurrentTaskHandle();
    g_fn = fn;
    g_ctx = ctx;
    g_i0 = mid;
    g_i1 = n;
    __sync_synchronize();
    xTaskNotifyGive(s_worker);
    fn(0, mid, ctx);
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    __sync_synchronize();
    prof_add(fn, esp_timer_get_time() - t0);
}

#ifndef RF_PIE_P4
/* 重みステージング: 上流の weak 既定（同期 memcpy）と同じ。時間だけ集計する。
 * PIE 版では pf_pie_linear.c の rf_stage_start（転置配置でステージ）がこれを置き換える */
void rf_stage_start(int slot, const void *src, size_t n) {
    int64_t t0 = esp_timer_get_time();
    memcpy(rf_stage_slot(slot), src, n);
    pf_stage_us += esp_timer_get_time() - t0;
    pf_stage_calls++;
    pf_stage_bytes += n;
}
#endif

void pf_par_reset(void) {
    s_nprof = 0;
    pf_stage_us = 0;
    pf_stage_calls = 0;
    pf_stage_bytes = 0;
}

void pf_par_report(void) {
    printf("  phase              calls      ms\n");
    for (int i = 0; i < s_nprof; i++)
        printf("  fn@%08x %8u %8.1f\n", (unsigned)(uintptr_t)s_prof[i].fn, (unsigned)s_prof[i].calls,
               s_prof[i].us / 1000.0);
    printf("  stage(memcpy)  %8u %8.1f  (%u KB)\n", (unsigned)pf_stage_calls, pf_stage_us / 1000.0,
           (unsigned)(pf_stage_bytes / 1024));
}
