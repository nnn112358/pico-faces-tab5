/* 密な行列積の PIE 版（列方向カーネル）と、そのための転置ステージング。
 *
 * 上流 kernels_ref.c の rf_linear2_i8 / rf_relu2sq2_i8 / rf_linear2_i8_acc16（と 1 トークン版）は
 * 出力 1 本ごとに内積（XACC）を取るので、内積の初期化と読み出しが MAC 本体と同じくらい重い。
 * 代わりに重みを転置して WT[k][o] とし、x[k] を 16 レーンにブロードキャストして
 *   QACC[o..o+16) += WT[k][o..o+16) · x[k]   （esp.vmulas.s8.qacc.ld.xp + esp.vldbc.8.ip）
 * を k について回すと、16 出力ぶんが K 命令 × 2 で終わる（esp-dl の s8 MatMul と同じ形）。
 *
 * 転置は blob から一度だけ PSRAM に作り（pf_stage_prepare）、rf_stage_start（weak を置き換え）が
 * EMB / QKV / PROJ / FC1 のスロットへ**転置済みの配置で** memcpy する。FINAL と FC2 は元の配置のまま
 * （final_range が rf_dot_i8 で行を直接読む / FC2 はもともと [K][O]）。
 * dit.c から arena のポインタを受けた wrap 版カーネルは、ポインタがどのスロットかで配置を判断する。
 * PIE が無効（自己テスト失敗）なら元の配置でステージし、wrap 版は参照実装へ落ちる。
 *
 * 結果は参照と bit 一致（int32 の加算順序に依らない。起動時の自己テストと golden CRC で確認）。 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "rf_model.h"
#include "rf_ops.h"

extern int64_t pf_stage_us;      /* pf_par.c の集計（プロファイル表示用） */
extern uint32_t pf_stage_calls;
extern size_t pf_stage_bytes;

typedef struct {
    const void *src;
    int8_t *T;
    size_t n;
    int K, O;
} tcopy_t;
#define PF_T_MAX 64
static tcopy_t s_t[PF_T_MAX];
static int s_nt;
static int s_slot_is_T[RF_SLOT_N];

/* DiT の重みステージング先。上流は PSRAM に置いた rf_arena のスロットを使うが、PIE のロードは
 * L1 ミス時に L2（PSRAM バック）のレイテンシで止まる（実測: ブロック転置でも 1 行 19 ns、
 * L1 に載っていれば 6.4 ns）。内部 SRAM の専用バッファに置き換える。
 * スロットの相対配置は上流と同じ（dit.c が FC2 = FC1 + 4·D·D を前提にしている）。
 * VAE デコーダの疎な重み（最大 147 KB）とは時間的に重ならないので同じ領域を共用する。 */
#define PF_STAGE_BYTES (2 * 4 * RF_DIM * RF_DIM + 3 * RF_DIM * RF_DIM + RF_DIM * RF_DIM + 2 * RF_DIM * RF_PD)
static int8_t *s_stage;            /* 内部 SRAM（16 整列）。無ければ rf_arena のスロット */

static size_t slot_off(int slot) {
    switch (slot) {
    case RF_SLOT_FC1:  return 0;
    case RF_SLOT_FC2:  return (size_t)4 * RF_DIM * RF_DIM;
    case RF_SLOT_QKV:  return (size_t)8 * RF_DIM * RF_DIM;
    case RF_SLOT_PROJ: return (size_t)11 * RF_DIM * RF_DIM;
    case RF_SLOT_EMB:  return (size_t)12 * RF_DIM * RF_DIM;
    default:           return (size_t)12 * RF_DIM * RF_DIM + (size_t)RF_DIM * RF_PD;
    }
}

static int8_t *slot_ptr(int slot) { return s_stage ? s_stage + slot_off(slot) : rf_stage_slot(slot); }

/* デコーダと共用するための入口 */
int8_t *pf_stage_buffer(size_t *len) {
    if (len) *len = s_stage ? (size_t)PF_STAGE_BYTES : 0;
    return s_stage;
}

static int slot_dims(int slot, size_t n, int *K, int *O) {
    switch (slot) {
    case RF_SLOT_EMB:  *K = RF_PD;  *O = RF_DIM;     return n == (size_t)RF_PD * RF_DIM;
    case RF_SLOT_QKV:  *K = RF_DIM; *O = 3 * RF_DIM; return n == (size_t)3 * RF_DIM * RF_DIM;
    case RF_SLOT_PROJ: *K = RF_DIM; *O = RF_DIM;     return n == (size_t)RF_DIM * RF_DIM;
    case RF_SLOT_FC1:  *K = RF_DIM; *O = 4 * RF_DIM; return n == (size_t)4 * RF_DIM * RF_DIM;
    default: return 0;
    }
}

static tcopy_t *find_T(const void *src) {
    for (int i = 0; i < s_nt; i++)
        if (s_t[i].src == src) return &s_t[i];
    return NULL;
}

/* W[O][K] → ブロック転置 T[O/16][K][16]（16 整列、PSRAM。末尾に 16 B の余白）。既にあればそれを返す。
 *   T[((og·K) + k)·16 + j] = W[(og·16 + j)·K + k]
 * 16 出力ぶんの K 行が連続するので、カーネルは +16 の連続ロードで済む（1 行ごとに別の
 * キャッシュラインを引く [K][O] 配置は、ロードが L1 に当たらず 1 行 20 ns かかった） */
static tcopy_t *make_T(const void *src, size_t n, int K, int O) {
    tcopy_t *t = find_T(src);
    if (t) return t;
    if (s_nt >= PF_T_MAX || (O & 15) || (K & 15) || K < 16) return NULL;
    int8_t *T = heap_caps_aligned_alloc(16, n + 16, MALLOC_CAP_SPIRAM);
    if (!T) return NULL;
    const int8_t *W = src;
    for (int og = 0; og < O / 16; og++)
        for (int k = 0; k < K; k++)
            for (int j = 0; j < 16; j++)
                T[((size_t)og * K + k) * 16 + j] = W[(size_t)(og * 16 + j) * K + k];
    memset(T + n, 0, 16);
    t = &s_t[s_nt++];
    t->src = src;
    t->T = T;
    t->n = n;
    t->K = K;
    t->O = O;
    return t;
}

/* 起動時に全ブロックの転置を作る（1.5 MB ほど PSRAM を使う）。戻り値 = 作った数 */
int pf_stage_prepare(const rf_model_t *m) {
    int made = 0;
    if (!s_stage) {
        s_stage = heap_caps_aligned_alloc(16, PF_STAGE_BYTES, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        printf("PIE: DiT weight staging buffer %u B in %s\n", (unsigned)PF_STAGE_BYTES,
               s_stage ? "internal SRAM" : "(alloc failed -> rf_arena in PSRAM)");
    }
    if (make_T(m->W_emb, (size_t)RF_PD * RF_DIM, RF_PD, RF_DIM)) made++;
    for (uint32_t b = 0; b < m->depth; b++) {
        const rf_blk_t *blk = &m->blk[b];
        if (blk->has_attn) {
            if (make_T(blk->Wqkv, (size_t)3 * RF_DIM * RF_DIM, RF_DIM, 3 * RF_DIM)) made++;
            if (make_T(blk->Wproj, (size_t)RF_DIM * RF_DIM, RF_DIM, RF_DIM)) made++;
        }
        if (make_T(blk->Wfc1, (size_t)4 * RF_DIM * RF_DIM, RF_DIM, 4 * RF_DIM)) made++;
    }
    return made;
}

/* weak な同期ステージングの置き換え（上流と同じ memcpy。転置版があればそれを置く） */
void rf_stage_start(int slot, const void *src, size_t n) {
    int64_t t0 = esp_timer_get_time();
    int K, O;
    const void *from = src;
    int isT = 0;
    if (rf_pie_enabled && slot_dims(slot, n, &K, &O)) {
        tcopy_t *t = find_T(src);
        if (t && t->n == n) {
            from = t->T;
            isT = 1;
        }
    }
    memcpy(slot_ptr(slot), from, n);
    s_slot_is_T[slot] = isT;
    pf_stage_us += esp_timer_get_time() - t0;
    pf_stage_calls++;
    pf_stage_bytes += n;
}

/* weak な rf_stage_wait の置き換え: 同期コピーなので待ちは無く、スロットのポインタを返す */
const int8_t *rf_stage_wait(int slot) { return slot_ptr(slot); }
void rf_stage_drain(void) {}

/* W が転置済みスロットを指していれば 1 */
static int w_is_T(const int8_t *W) {
    for (int slot = 0; slot < RF_SLOT_N; slot++)
        if (W == slot_ptr(slot)) return s_slot_is_T[slot];
    return 0;
}

/* ---- 列方向カーネル: raw[0..16) = Σ_k WB[k·16 + 0..16) · x[k]。K は 16 の倍数、WB は 16 整列 ----
 * WB はブロック転置の 1 グループ（K 行 × 16 B が連続）。x は 16 個ずつ q7 に読み、
 *   esp.vsmulas.s8.qacc.ld.incp qu, rs, qx, qy, sel   （QACC += qx · qy[sel]、qu ← [rs]、rs += 16）
 * で「行 k の MAC（スカラは q7 のレーン sel）」と「行 k+1 の連続ロード」を 1 命令にする。
 * 1 チャンク（16 行）= vld 1 + 融合 16 = PIE 命令 17 個。行あたり ≈ 6.4 ns（実測の命令コスト）。
 * 最後の融合命令が行 K（ブロックの 1 行先）を読むので、転置バッファは末尾に 16 B の余白を持つ。 */
#define VS(qu, qx, sel) "  esp.vsmulas.s8.qacc.ld.incp " qu ", %[pw], " qx ", q7, " sel "\n"
static inline void pf_colmac16(const int8_t *wb, const int8_t *x, int K, int32_t *raw16) {
    register const int8_t *pw __asm__("a2") = wb;
    register const int8_t *px __asm__("a3") = x;
    register int k __asm__("a4") = K >> 4;
    register int32_t *out __asm__("t3") = raw16;
    __asm__ volatile(
        ".option push                                            \n"
        ".option arch, +xesppie                                  \n"
        "esp.zero.qacc                                           \n"
        "esp.vld.128.ip q0, %[pw], 16                            \n"
        "1:                                                      \n"
        "  esp.vld.128.ip q7, %[px], 16                          \n"
        VS("q1", "q0", "0")  VS("q0", "q1", "1")  VS("q1", "q0", "2")  VS("q0", "q1", "3")
        VS("q1", "q0", "4")  VS("q0", "q1", "5")  VS("q1", "q0", "6")  VS("q0", "q1", "7")
        VS("q1", "q0", "8")  VS("q0", "q1", "9")  VS("q1", "q0", "10") VS("q0", "q1", "11")
        VS("q1", "q0", "12") VS("q0", "q1", "13") VS("q1", "q0", "14") VS("q0", "q1", "15")
        "  addi %[k], %[k], -1                                   \n"
        "  bnez %[k], 1b                                         \n"
        "esp.st.qacc.l.l.128.ip %[out], 16                       \n"
        "esp.st.qacc.l.h.128.ip %[out], 16                       \n"
        "esp.st.qacc.h.l.128.ip %[out], 16                       \n"
        "esp.st.qacc.h.h.128.ip %[out], 16                       \n"
        ".option pop                                             \n"
        : [pw] "+r"(pw), [px] "+r"(px), [k] "+r"(k), [out] "+r"(out)
        :
        : "memory");
}
#undef VS

static inline int8_t sq_rq(int32_t a, int32_t M, uint8_t s) { /* kernels_ref.c の rf_sq_rq と同じ */
    if (a <= 0) return 0;
    int64_t u = ((int64_t)a * a) >> 12;
    return rf_sat8(rf_rq(u, M, s));
}

/* 3 種類のエピローグ */
enum { EP_I8 = 0, EP_I8_RELU = 1, EP_SQ = 2, EP_ACC16 = 3 };

static void colmat(const int8_t *x, const int8_t *WT, const int32_t *b, const int32_t *M, const uint8_t *s,
                   int K, int O, int ep, int8_t *y, int16_t *res) {
    for (int og = 0; og < O; og += 16) {
        int32_t raw[16] __attribute__((aligned(16)));
        pf_colmac16(WT + (size_t)og * K, x, K, raw); /* グループ og/16 のブロック先頭 = (og/16)·K·16 */
        for (int j = 0; j < 16; j++) {
            int o = og + j;
            int32_t acc = (int32_t)((uint32_t)b[o] + (uint32_t)raw[j]);
            switch (ep) {
            case EP_SQ: y[o] = sq_rq(acc, M[o], s[o]); break;
            case EP_ACC16: res[o] = rf_sat16((int64_t)res[o] + rf_rq(acc, M[o], s[o])); break;
            default: {
                int64_t v = rf_rq(acc, M[o], s[o]);
                if (ep == EP_I8_RELU && v < 0) v = 0;
                y[o] = rf_sat8(v);
            }
            }
        }
    }
}

/* 自己テスト用の入口（WT はブロック転置 [O/16][K][16] + 16 B 余白を呼び出し側が用意する） */
void pf_pie_colmat_test(const int8_t *x, const int8_t *WT, const int32_t *b, const int32_t *M, const uint8_t *s,
                        int K, int O, int ep, int8_t *y, int16_t *res) {
    colmat(x, WT, b, M, s, K, O, ep, y, res);
}

/* ---- wrap 版 ------------------------------------------------------------------------- */
void __real_rf_linear_i8(const int8_t *x, const int8_t *W, const int32_t *b, const int32_t *M, const uint8_t *s,
                         int K, int O, int relu, int8_t *y);
void __real_rf_linear2_i8(const int8_t *x, const int8_t *W, const int32_t *b, const int32_t *M, const uint8_t *s,
                          int K, int O, int relu, int8_t *y0, int8_t *y1);
void __real_rf_relu2sq_i8(const int8_t *x, const int8_t *W, const int32_t *b, const int32_t *M, const uint8_t *s,
                          int K, int O, int8_t *y);
void __real_rf_relu2sq2_i8(const int8_t *x, const int8_t *W, const int32_t *b, const int32_t *M, const uint8_t *s,
                           int K, int O, int8_t *y0, int8_t *y1);
void __real_rf_linear_i8_acc16(const int8_t *x, const int8_t *W, const int32_t *b, const int32_t *M,
                               const uint8_t *s, int K, int O, int16_t *res);
void __real_rf_linear2_i8_acc16(const int8_t *x, const int8_t *W, const int32_t *b, const int32_t *M,
                                const uint8_t *s, int K, int O, int16_t *res0, int16_t *res1);

#define USABLE(W, K, O) (rf_pie_enabled && ((K) & 15) == 0 && (K) >= 16 && ((O) & 15) == 0 && w_is_T(W))

void __wrap_rf_linear_i8(const int8_t *x, const int8_t *W, const int32_t *b, const int32_t *M, const uint8_t *s,
                         int K, int O, int relu, int8_t *y) {
    if (!USABLE(W, K, O)) { __real_rf_linear_i8(x, W, b, M, s, K, O, relu, y); return; }
    colmat(x, W, b, M, s, K, O, relu ? EP_I8_RELU : EP_I8, y, NULL);
}

void __wrap_rf_linear2_i8(const int8_t *x, const int8_t *W, const int32_t *b, const int32_t *M, const uint8_t *s,
                          int K, int O, int relu, int8_t *y0, int8_t *y1) {
    if (!USABLE(W, K, O)) { __real_rf_linear2_i8(x, W, b, M, s, K, O, relu, y0, y1); return; }
    colmat(x, W, b, M, s, K, O, relu ? EP_I8_RELU : EP_I8, y0, NULL);
    colmat(x + K, W, b, M, s, K, O, relu ? EP_I8_RELU : EP_I8, y1, NULL);
}

void __wrap_rf_relu2sq_i8(const int8_t *x, const int8_t *W, const int32_t *b, const int32_t *M, const uint8_t *s,
                          int K, int O, int8_t *y) {
    if (!USABLE(W, K, O)) { __real_rf_relu2sq_i8(x, W, b, M, s, K, O, y); return; }
    colmat(x, W, b, M, s, K, O, EP_SQ, y, NULL);
}

void __wrap_rf_relu2sq2_i8(const int8_t *x, const int8_t *W, const int32_t *b, const int32_t *M, const uint8_t *s,
                           int K, int O, int8_t *y0, int8_t *y1) {
    if (!USABLE(W, K, O)) { __real_rf_relu2sq2_i8(x, W, b, M, s, K, O, y0, y1); return; }
    colmat(x, W, b, M, s, K, O, EP_SQ, y0, NULL);
    colmat(x + K, W, b, M, s, K, O, EP_SQ, y1, NULL);
}

void __wrap_rf_linear_i8_acc16(const int8_t *x, const int8_t *W, const int32_t *b, const int32_t *M,
                               const uint8_t *s, int K, int O, int16_t *res) {
    if (!USABLE(W, K, O)) { __real_rf_linear_i8_acc16(x, W, b, M, s, K, O, res); return; }
    colmat(x, W, b, M, s, K, O, EP_ACC16, NULL, res);
}

void __wrap_rf_linear2_i8_acc16(const int8_t *x, const int8_t *W, const int32_t *b, const int32_t *M,
                                const uint8_t *s, int K, int O, int16_t *res0, int16_t *res1) {
    if (!USABLE(W, K, O)) { __real_rf_linear2_i8_acc16(x, W, b, M, s, K, O, res0, res1); return; }
    colmat(x, W, b, M, s, K, O, EP_ACC16, NULL, res0);
    colmat(x + K, W, b, M, s, K, O, EP_ACC16, NULL, res1);
}
