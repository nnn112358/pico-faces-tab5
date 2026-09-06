/* PIE 版の VAE デコーダ（疎な 3×3 畳み込みを QACC で累算する）。
 *
 * 上流 vae_dec.c の rf_decode は dit.c から呼ばれる（別の翻訳単位）ので、リンカの
 * --wrap=rf_decode でこちらの __wrap_ 版に差し替える。上流のソースは無改変。
 * 参照実装は __real_rf_decode として残り、準備に失敗したときと PIE が無効のときに使う。
 *
 * 構造は上流 vae_dec.c と同じ（unpatchify → requant → 畳み込みの連鎖を rf_arena で ping-pong）。
 * 違いは疎な層（flags bit 3）の内側だけ:
 *   参照: 出力画素ごとに acc[O] = b、非ゼロ (ch, val) ごとに acc[0..O) += val · W[tap][ch][0..O)
 *   PIE : 16 出力ごとに QACC を 0 にし、非ゼロごとに
 *           esp.vld.128.ip   q0, wrow, 0     W[tap][ch][og·16 .. +16)（O=8 の層は esp.vld.l.64.ip）
 *           esp.vldbc.8.ip   q1, pval, 1     val をブロードキャスト
 *           esp.vmulas.s8.qacc q0, q1        16 レーン × 32 bit に累算
 *         最後に QACC を書き出して b を足し、rq → ReLU → sat8（参照と同じ式）
 *   int32 の加算は順序に依らないので結果は bit 一致する（自己テストと golden CRC で確認）。
 *
 * 重みの整列: blob の中の W は 4 バイト境界にしかない（実測 0/4/8/12）。PIE の 128 bit ロードは
 * 16 バイト境界が要るので、層ごとに内部 RAM の 16 整列バッファ（最大層 = 9·128·128 = 147,456 B）へ
 * memcpy してから使う。行 (tap, ch) の先頭 = base + ((tap·C + ch) · O) は O ≥ 16 なら 16 の倍数、
 * O = 8 なら 8 の倍数なので、O = 8 の層は 64 bit ロード（レーン 8–15 は捨てる）。
 * ⚠️ QACC は 1 本しか無いので、出力 16 本ごとに非ゼロのリストを歩き直す（O=128 なら 8 回）。
 *    それでも 1 命令 16 MAC なのでスカラ（1 MAC ≈ 6〜8 命令）より十分速い。 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "rf_model.h"
#include "rf_ops.h"

#define MAX_ROW_NZ RF_DEC_NZ_MAX
#define MAX_SP_O RF_DEC_O_MAX
#define MAX_SP_W RF_DEC_W_MAX

typedef struct {
    int yi;
    uint16_t start[MAX_SP_W + 1];
    uint8_t ch[MAX_ROW_NZ];
    int8_t val[MAX_ROW_NZ];
} rowcomp_t;

/* 上流と同じく、DiT の qkv/attn バッファ（rf_decode 中は遊んでいる）に行圧縮の作業域を重ねる */
extern int8_t *const rf_dec_scratch;
#define rowbuf ((rowcomp_t(*)[3])rf_dec_scratch)
_Static_assert(RF_TOKENS * 4 * RF_DIM >= (int)(2 * 3 * sizeof(rowcomp_t)),
               "rowbuf exceeds DiT qkv/attn scratch");

void __real_rf_decode(const rf_model_t *m, const int16_t z_tok[RF_TOKENS][RF_PD], uint8_t *img);
int8_t *pf_stage_buffer(size_t *len);

int64_t pf_vae_us;               /* 直近の rf_decode にかかった時間（main が読む） */
int pf_vae_pie;                  /* 直近の rf_decode が PIE 経路だったか */

static int8_t *s_wbuf;           /* 16 整列の重みバッファ（最大の疎な層ぶん） */
static size_t s_wbuf_len;
static const rf_model_t *s_prepared;
static int16_t s_zhwc[RF_ZHW * RF_ZHW * RF_ZCH] __attribute__((aligned(16)));

typedef struct {
    const int8_t *x;
    const rf_declayer_t *L;
    const int8_t *Wal;           /* 16 整列の重み（疎な層） */
    int H, W;
    int relu, up, u8, wt;
    int sh;                      /* log2(O) */
    int o8;                      /* O == 8 → 64 bit ロード */
    int8_t *dst;
} dctx_t;

static void compact_row(const dctx_t *c, rowcomp_t *rc, int yi) {
    const int C = (int)c->L->C;
    const int8_t *row = c->x + (size_t)yi * c->W * C;
    int m = 0;
    for (int xi = 0; xi < c->W; xi++) {
        rc->start[xi] = (uint16_t)m;
        for (int ch = 0; ch < C; ch++) {
            int8_t v = row[xi * C + ch];
            if (v) {
                rc->ch[m] = (uint8_t)ch;
                rc->val[m] = v;
                m++;
            }
        }
    }
    rc->start[c->W] = (uint16_t)m;
    rc->yi = yi;
}

/* ---- PIE プリミティブ（QACC は asm ブロックをまたいで生きる。volatile asm は並べ替えられない）---- */
static inline void pf_qacc_zero(void) {
    __asm__ volatile(".option push\n.option arch, +xesppie\nesp.zero.qacc\n.option pop\n" ::: "memory");
}

static inline void pf_qacc_store(int32_t *raw16) {
    register int32_t *out __asm__("a2") = raw16;
    __asm__ volatile(
        ".option push                          \n"
        ".option arch, +xesppie                \n"
        "esp.st.qacc.l.l.128.ip %[out], 16     \n"
        "esp.st.qacc.l.h.128.ip %[out], 16     \n"
        "esp.st.qacc.h.l.128.ip %[out], 16     \n"
        "esp.st.qacc.h.h.128.ip %[out], 16     \n"
        ".option pop                           \n"
        : [out] "+r"(out)
        :
        : "memory");
}

/* QACC[0..16) += Σ_{i<n} val[i] · wtap[(ch[i] << sh) + 0..16)。n ≥ 1、wtap は 16 整列。
 * パイプライン: 次の行アドレスを先に計算し、融合命令で「今の MAC」と「次の行のロード」を同時に出す */
static inline void pf_gather16(const uint8_t *ch, const int8_t *val, int n, const int8_t *wtap, int sh) {
    register const uint8_t *pch __asm__("a2") = ch;
    register const int8_t *pval __asm__("a3") = val;
    register const int8_t *wrow __asm__("a4");
    register const int8_t *wb __asm__("a5") = wtap;
    register int k __asm__("t3") = n - 1;
    register int shv __asm__("t4") = sh;
    register int zero __asm__("t5") = 0;
    __asm__ volatile(
        ".option push                                          \n"
        ".option arch, +xesppie                                \n"
        "lbu   %[wrow], 0(%[pch])                              \n"
        "sll   %[wrow], %[wrow], %[shv]                        \n"
        "add   %[wrow], %[wrow], %[wb]                         \n"
        "esp.vld.128.ip q0, %[wrow], 0                         \n"
        "esp.vldbc.8.ip q1, %[pval], 1                         \n"
        "addi  %[pch], %[pch], 1                               \n"
        "beqz  %[k], 5f                                        \n"
        "1:                                                    \n"
        "  lbu   %[wrow], 0(%[pch])                            \n"
        "  sll   %[wrow], %[wrow], %[shv]                      \n"
        "  add   %[wrow], %[wrow], %[wb]                       \n"
        "  esp.vmulas.s8.qacc.ld.xp q2, %[wrow], %[z], q0, q1  \n"
        "  esp.vldbc.8.ip q3, %[pval], 1                       \n"
        "  addi  %[pch], %[pch], 1                             \n"
        "  addi  %[k], %[k], -1                                \n"
        "  beqz  %[k], 3f                                      \n"
        "  lbu   %[wrow], 0(%[pch])                            \n"
        "  sll   %[wrow], %[wrow], %[shv]                      \n"
        "  add   %[wrow], %[wrow], %[wb]                       \n"
        "  esp.vmulas.s8.qacc.ld.xp q0, %[wrow], %[z], q2, q3  \n"
        "  esp.vldbc.8.ip q1, %[pval], 1                       \n"
        "  addi  %[pch], %[pch], 1                             \n"
        "  addi  %[k], %[k], -1                                \n"
        "  bnez  %[k], 1b                                      \n"
        "5:                                                    \n"
        "  esp.vmulas.s8.qacc q0, q1                           \n"
        "  j 4f                                                \n"
        "3:                                                    \n"
        "  esp.vmulas.s8.qacc q2, q3                           \n"
        "4:                                                    \n"
        ".option pop                                           \n"
        : [pch] "+r"(pch), [pval] "+r"(pval), [wrow] "=&r"(wrow), [k] "+r"(k)
        : [wb] "r"(wb), [shv] "r"(shv), [z] "r"(zero)
        : "memory");
}

/* O = 8 用: 行は 8 バイト（8 整列）。下位 64 bit だけ読む（融合命令には 64 bit 版が無いので分離）。
 * レーン 8–15 は不定（捨てる）。同じく次の行を先に読む */
static inline void pf_gather8(const uint8_t *ch, const int8_t *val, int n, const int8_t *wtap) {
    register const uint8_t *pch __asm__("a2") = ch;
    register const int8_t *pval __asm__("a3") = val;
    register const int8_t *wrow __asm__("a4");
    register const int8_t *wb __asm__("a5") = wtap;
    register int k __asm__("t3") = n - 1;
    __asm__ volatile(
        ".option push                                          \n"
        ".option arch, +xesppie                                \n"
        "lbu   %[wrow], 0(%[pch])                              \n"
        "slli  %[wrow], %[wrow], 3                             \n"
        "add   %[wrow], %[wrow], %[wb]                         \n"
        "esp.vld.l.64.ip q0, %[wrow], 0                        \n"
        "esp.vldbc.8.ip q1, %[pval], 1                         \n"
        "addi  %[pch], %[pch], 1                               \n"
        "beqz  %[k], 5f                                        \n"
        "1:                                                    \n"
        "  lbu   %[wrow], 0(%[pch])                            \n"
        "  slli  %[wrow], %[wrow], 3                           \n"
        "  add   %[wrow], %[wrow], %[wb]                       \n"
        "  esp.vld.l.64.ip q2, %[wrow], 0                      \n"
        "  esp.vldbc.8.ip q3, %[pval], 1                       \n"
        "  esp.vmulas.s8.qacc q0, q1                           \n"
        "  addi  %[pch], %[pch], 1                             \n"
        "  addi  %[k], %[k], -1                                \n"
        "  beqz  %[k], 3f                                      \n"
        "  lbu   %[wrow], 0(%[pch])                            \n"
        "  slli  %[wrow], %[wrow], 3                           \n"
        "  add   %[wrow], %[wrow], %[wb]                       \n"
        "  esp.vld.l.64.ip q0, %[wrow], 0                      \n"
        "  esp.vldbc.8.ip q1, %[pval], 1                       \n"
        "  esp.vmulas.s8.qacc q2, q3                           \n"
        "  addi  %[pch], %[pch], 1                             \n"
        "  addi  %[k], %[k], -1                                \n"
        "  bnez  %[k], 1b                                      \n"
        "5:                                                    \n"
        "  esp.vmulas.s8.qacc q0, q1                           \n"
        "  j 4f                                                \n"
        "3:                                                    \n"
        "  esp.vmulas.s8.qacc q2, q3                           \n"
        "4:                                                    \n"
        ".option pop                                           \n"
        : [pch] "+r"(pch), [pval] "+r"(pval), [wrow] "=&r"(wrow), [k] "+r"(k)
        : [wb] "r"(wb)
        : "memory");
}

/* 自己テスト用の入口: acc[0..lanes) = Σ val[i] · W[(ch[i] << sh) + 0..lanes) */
void pf_pie_gather_test(const uint8_t *ch, const int8_t *val, int n, const int8_t *W, int O, int32_t *acc) {
    int32_t raw[16] __attribute__((aligned(16)));
    pf_qacc_zero();
    if (O == 8) pf_gather8(ch, val, n, W);
    else {
        int sh = 0;
        while ((1 << sh) < O) sh++;
        pf_gather16(ch, val, n, W, sh);
    }
    pf_qacc_store(raw);
    memcpy(acc, raw, (size_t)(O == 8 ? 8 : 16) * sizeof(int32_t));
}

static void conv_rows_sparse_pie(int y0, int y1, void *p) {
    dctx_t *c = p;
    const rf_declayer_t *L = c->L;
    const int C = (int)L->C, O = (int)L->O;
    const int Wo = c->up ? 2 * c->W : c->W;
    const int ngroups = c->o8 ? 1 : O / 16;
    const int lanes = c->o8 ? 8 : 16;
    rowcomp_t *slots = rowbuf[rf_core_id()];
    for (int i = 0; i < 3; i++) slots[i].yi = -2;

    for (int yo = y0; yo < y1; yo++) {
        for (int xo = 0; xo < Wo; xo++) {
            int8_t *out = c->dst + ((size_t)yo * Wo + xo) * O;
            for (int og = 0; og < ngroups; og++) {
                pf_qacc_zero();
                for (int dy = 0; dy < 3; dy++) {
                    int yi = yo + dy - 1;
                    if (c->up) yi >>= 1;
                    if (yi < 0 || yi >= c->H) continue;
                    rowcomp_t *rc = &slots[yi % 3];
                    if (rc->yi != yi) compact_row(c, rc, yi);
                    for (int dx = 0; dx < 3; dx++) {
                        int xi = xo + dx - 1;
                        if (c->up) xi >>= 1;
                        if (xi < 0 || xi >= c->W) continue;
                        int i0 = rc->start[xi], n = rc->start[xi + 1] - i0;
                        if (!n) continue;
                        const int8_t *wtap = c->Wal + ((size_t)(dy * 3 + dx) * C) * O + og * 16;
                        if (c->o8) pf_gather8(rc->ch + i0, rc->val + i0, n, wtap);
                        else pf_gather16(rc->ch + i0, rc->val + i0, n, wtap, c->sh);
                    }
                }
                int32_t raw[16] __attribute__((aligned(16)));
                pf_qacc_store(raw);
                for (int j = 0; j < lanes; j++) {
                    int o = og * 16 + j;
                    int32_t acc = (int32_t)((uint32_t)L->b[o] + (uint32_t)raw[j]);
                    int64_t v = rf_rq(acc, L->M[o], L->s[o]);
                    if (v < 0) v = 0; /* 疎な層は必ず conv+ReLU（上流と同じ） */
                    out[o] = rf_sat8(v);
                }
            }
        }
    }
}

static void conv_rows_dense(int y0, int y1, void *p) {
    dctx_t *c = p;
    rf_conv3x3_i8_rows(c->x, c->H, c->W, (int)c->L->C, c->L->W, c->L->b, c->L->M, c->L->s,
                       (int)c->L->O, c->relu, 1, c->up, c->u8, c->dst, NULL, y0, y1);
}

/* 疎な層の条件を確かめ、重みバッファを確保する。戻り値 1 = PIE 経路を使える */
static int pf_dec_prepare(const rf_model_t *m) {
    if (s_prepared == m) return s_wbuf != NULL;
    s_prepared = m;
    size_t need = 0;
    for (uint32_t i = 0; i < m->n_dec; i++) {
        const rf_declayer_t *L = &m->dec[i];
        if (!((L->flags >> 3) & 1)) continue;
        int O = (int)L->O;
        if (!(O == 8 || O == 16 || O == 32 || O == 64 || O == 128) || O > MAX_SP_O || L->C > 255) {
            printf("PIE decode: dec[%u] C=%u O=%u unsupported -> reference decoder\n", (unsigned)i,
                   (unsigned)L->C, (unsigned)L->O);
            return 0;
        }
        size_t sz = (size_t)9 * L->C * L->O;
        if (sz > need) need = sz;
    }
    if (!need) return 0;
    s_wbuf_len = need + 16;
    /* DiT の重みステージング用の内部バッファ（pf_pie_linear.c）と時間的に重ならないので共用する */
    size_t shared_len = 0;
    int8_t *shared = pf_stage_buffer(&shared_len);
    if (shared && shared_len >= s_wbuf_len) {
        s_wbuf = shared;
        printf("PIE decode: weight buffer %u B shares the DiT staging buffer (internal)\n", (unsigned)s_wbuf_len);
        return 1;
    }
    s_wbuf = heap_caps_aligned_alloc(16, s_wbuf_len, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    const char *where = "internal";
    if (!s_wbuf) {
        s_wbuf = heap_caps_aligned_alloc(16, s_wbuf_len, MALLOC_CAP_SPIRAM);
        where = "PSRAM";
    }
    if (!s_wbuf) {
        printf("PIE decode: weight buffer %u B alloc failed -> reference decoder\n", (unsigned)s_wbuf_len);
        return 0;
    }
    printf("PIE decode: weight buffer %u B in %s\n", (unsigned)s_wbuf_len, where);
    return 1;
}

void __wrap_rf_decode(const rf_model_t *m, const int16_t z_tok[RF_TOKENS][RF_PD], uint8_t *img) {
    int64_t t0 = esp_timer_get_time();
    if (!rf_pie_enabled || !pf_dec_prepare(m)) {
        __real_rf_decode(m, z_tok, img);
        pf_vae_us = esp_timer_get_time() - t0;
        pf_vae_pie = 0;
        return;
    }
    /* unpatchify（上流と同じ） */
    int16_t *zhwc = s_zhwc;
    const int P = RF_PATCH, G = RF_ZHW / RF_PATCH;
    for (int y = 0; y < RF_ZHW; y++)
        for (int x = 0; x < RF_ZHW; x++)
            for (int c = 0; c < RF_ZCH; c++)
                zhwc[(y * RF_ZHW + x) * RF_ZCH + c] =
                    z_tok[(y / P) * G + (x / P)][c * P * P + (y % P) * P + (x % P)];
    rf_requant_i16_to_i8(zhwc, RF_ZHW * RF_ZHW * RF_ZCH, m->M_zdec, m->s_zdec, rf_arena[0]);

    int H = RF_ZHW, W = RF_ZHW;
    const int8_t *in = rf_arena[0];
    int8_t *out = rf_arena[1];
    for (uint32_t i = 0; i < m->n_dec; i++) {
        const rf_declayer_t *L = &m->dec[i];
        dctx_t c = {0};
        c.x = in;
        c.L = L;
        c.H = H;
        c.W = W;
        c.up = (int)(L->flags & 1);
        c.relu = (int)((L->flags >> 1) & 1);
        c.u8 = (int)((L->flags >> 2) & 1);
        c.wt = (int)((L->flags >> 3) & 1);
        c.dst = c.u8 ? (int8_t *)img : out;
        int Ho = c.up ? 2 * H : H;
        if (c.wt) {
            memcpy(s_wbuf, L->W, (size_t)9 * L->C * L->O);
            c.Wal = s_wbuf;
            c.o8 = (L->O == 8);
            c.sh = 0;
            while ((1u << c.sh) < L->O) c.sh++;
            rf_par_for(Ho, conv_rows_sparse_pie, &c);
        } else {
            rf_par_for(Ho, conv_rows_dense, &c);
        }
        if (c.up) {
            H *= 2;
            W *= 2;
        }
        in = c.dst;
        out = (in == (const int8_t *)rf_arena[0]) ? rf_arena[1] : rf_arena[0];
    }
    pf_vae_us = esp_timer_get_time() - t0;
    pf_vae_pie = 1;
}
