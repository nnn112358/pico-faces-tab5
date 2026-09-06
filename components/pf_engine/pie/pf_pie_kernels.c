/* PIE 版の疎な行列積（fc2）と、起動時の自己テスト。
 *
 * 上流 kernels_ref.c の rf_axpy_acc16_sp は dit.c から呼ばれる（別の翻訳単位）ので、
 * リンカの --wrap=rf_axpy_acc16_sp でこちらの __wrap_ 版に差し替える。上流のソースは無改変。
 * 参照実装は __real_rf_axpy_acc16_sp として残り、条件が合わないときや自己テストに落ちたときに使う。
 *
 * 数学（参照実装と同じ）:
 *   res[o] = sat16(res[o] + rq(b[o] + Σ_i val[i]·Wt[idx[i]][o], M[o], s[o]))
 * 参照は a[j] = b[j] から始めて axpy を足すが、こちらは 0 から累算して最後に b を足す。
 * int32 の加算は順序に依らないので結果は同じ。
 *
 * PIE の使い方: 16 出力（O=128 なら 8 グループ）ごとに QACC を 0 にし、非ゼロ i ごとに
 *   esp.vldbc.8.ip  qv, pval, 1     val[i] を 16 レーンにブロードキャスト
 *   esp.vld.128.ip  qw, wrow, 0     Wt[idx[i]][ob..ob+16)（16 整列: Wt は arena、O=128）
 *   esp.vmulas.s8.qacc qw, qv       16 レーンの積を QACC（16 × 32 bit）に加算
 * 最後に esp.st.qacc.{l,h}.{l,h}.128.ip で QACC を 64 B の配列に書き出す。
 * ⚠️ QACC のレーン順（どの 32 bit がどの出力か）は自己テストで実機確認する。
 *    esp-dl の資料は「S8 の MAC は QACC_L/H に 16 × 32 bit」としか書いていない。
 * ⚠️ vmulas.s8.qacc は 32 bit で飽和する。モデル設計で |Σ| < 2^31 なので飽和は起きない。 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "esp_timer.h"
#include "rf_ops.h"

int rf_pie_enabled = 1;
int rf_pie_unaligned_ok = 0; /* 自己テストが通るまで memcpy 経路 */

/* ---- rmsnorm（bit 一致の書き直し。dit.c からの呼び出しを --wrap で受ける）-------------
 * 参照（kernels_ref.c）:
 *   ss = Σ x²;  a8 = isqrt64((ss << 16) / K);  R = 2^38 / max(a8, 1)
 *   y = sat8((x·R·G + (B << 31) + 2^37) >> 38)
 * 変更点（どれも整数演算の結果を変えない）:
 *   - K が 2 のべき（32 / 128）なら (ss << 16) / K を右シフトに（符号なし除算 = シフト）
 *   - x·R·G を (x·G)·R の順に（オーバーフローしない範囲では積は結合的）。x·G は int32 に
 *     収まり、R < 2^31 なら 32×32→64 の乗算 1 回（mul + mulh）で済む。R ≥ 2^31（入力がほぼ
 *     ゼロのとき）は参照と同じ 64 bit 演算に落とす */
void __real_rf_rmsnorm_i16_to_i8(const int16_t *x, const int16_t *G, const int16_t *B, int K, int8_t *y);

/* floor(sqrt(v)) を厳密に。FPU（単精度）の推定を整数の比較で ±数回補正する。
 * v < 2^52 なら float の相対誤差 2^-24 は絶対誤差 1 未満に収まる。それ以上は参照のビットループ */
static uint32_t pf_isqrt64_fast(uint64_t v) {
    if (v >> 52) return rf_isqrt64(v);
    float f = (float)(uint32_t)(v >> 32) * 4294967296.0f + (float)(uint32_t)v;
    uint32_t r = (uint32_t)__builtin_sqrtf(f);
    while ((uint64_t)r * r > v) r--;
    while ((uint64_t)(r + 1) * (r + 1) <= v) r++;
    return r;
}

/* floor(2^38 / d) を厳密に、64 bit のソフト除算を使わずに。
 * 2^32 = q0·d + (r0 + 1)（q0 = (2^32−1)/d、r0 = (2^32−1) mod d）なので
 * 2^38 = 64·q0·d + 64·(r0+1) → floor = 64·q0 + floor(64·(r0+1)/d)。d < 2^26 なら 64·(r0+1) < 2^32 */
static int64_t pf_div238(uint32_t d) {
    if (d >= (1u << 26)) return ((int64_t)1 << (RMS_R_SHIFT + 8)) / d;
    uint32_t q0 = 0xFFFFFFFFu / d, r0 = 0xFFFFFFFFu % d;
    return ((int64_t)q0 << 6) + (int64_t)((64u * (r0 + 1)) / d);
}

void __wrap_rf_rmsnorm_i16_to_i8(const int16_t *x, const int16_t *G, const int16_t *B, int K, int8_t *y) {
    int64_t ss = 0;
    for (int k = 0; k < K; k++) ss += (int64_t)((int32_t)x[k] * x[k]);
    uint64_t v = (uint64_t)ss << 16;
    if (K > 0 && (K & (K - 1)) == 0) {
        int lg = 0;
        while ((1 << lg) < K) lg++;
        v >>= lg;
    } else {
        v /= (uint64_t)K;
    }
    uint32_t a8 = pf_isqrt64_fast(v);
    int64_t R = pf_div238(a8 ? a8 : 1);
    if (R < ((int64_t)1 << 31)) {
        int32_t R32 = (int32_t)R;
        for (int k = 0; k < K; k++) {
            int32_t xg = (int32_t)x[k] * G[k];
            int64_t p = (int64_t)xg * R32 + ((int64_t)B[k] << (RMS_OUT_SHIFT - RMS_B_Q));
            y[k] = rf_sat8((p + ((int64_t)1 << (RMS_OUT_SHIFT - 1))) >> RMS_OUT_SHIFT);
        }
    } else {
        for (int k = 0; k < K; k++) {
            int64_t p = (int64_t)x[k] * R * G[k] + ((int64_t)B[k] << (RMS_OUT_SHIFT - RMS_B_Q));
            y[k] = rf_sat8((p + ((int64_t)1 << (RMS_OUT_SHIFT - 1))) >> RMS_OUT_SHIFT);
        }
    }
}

/* QACC のレーン → 出力番号の対応。自己テストで確定する（既定は恒等） */
static uint8_t s_lane_of[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

void pf_pie_gather_test(const uint8_t *ch, const int8_t *val, int n, const int8_t *W, int O, int32_t *acc);
void pf_pie_colmat_test(const int8_t *x, const int8_t *WT, const int32_t *b, const int32_t *M, const uint8_t *s,
                        int K, int O, int ep, int8_t *y, int16_t *res);
void __real_rf_linear2_i8(const int8_t *x, const int8_t *W, const int32_t *b, const int32_t *M, const uint8_t *s,
                          int K, int O, int relu, int8_t *y0, int8_t *y1);
void __real_rf_relu2sq2_i8(const int8_t *x, const int8_t *W, const int32_t *b, const int32_t *M, const uint8_t *s,
                           int K, int O, int8_t *y0, int8_t *y1);
void __real_rf_linear2_i8_acc16(const int8_t *x, const int8_t *W, const int32_t *b, const int32_t *M,
                                const uint8_t *s, int K, int O, int16_t *res0, int16_t *res1);
void __real_rf_axpy_acc16_sp(const uint16_t *idx, const int8_t *val, int m,
                             const int8_t *Wt, int O, const int32_t *b,
                             const int32_t *M, const uint8_t *s, int16_t *res);

/* acc[16] = Σ_i val[i] · Wt[idx[i]*O + 0..16)。wbase = Wt + ob（16 整列）、O = 128 固定。
 * raw[16] は QACC を書き出したままの順（レーン順の解決は呼び出し側）。
 * パイプライン: 次の行アドレスを先に計算し、融合命令 esp.vmulas.s8.qacc.ld.xp で
 * 「今の MAC」と「次の行のロード」を同時に出す（(q0,q1) と (q2,q3) を交互に使う）。
 * m = 1 は単純経路。 */
static inline void pf_pie_axpy16_raw(const uint16_t *idx, const int8_t *val, int m,
                                     const int8_t *wbase, int32_t *raw16) {
    register const uint16_t *pidx __asm__("a2") = idx;
    register const int8_t *pval __asm__("a3") = val;
    register const int8_t *wrow __asm__("a4");
    register const int8_t *wb __asm__("a5") = wbase;
    register int k __asm__("t3") = m - 1;
    register int32_t *out __asm__("t4") = raw16;
    register int zero __asm__("t5") = 0;
    __asm__ volatile(
        ".option push                                          \n"
        ".option arch, +xesppie                                \n"
        "esp.zero.qacc                                         \n"
        "lhu   %[wrow], 0(%[pidx])                             \n"
        "slli  %[wrow], %[wrow], 7                             \n"
        "add   %[wrow], %[wrow], %[wb]                         \n"
        "esp.vld.128.ip q0, %[wrow], 0                         \n"
        "esp.vldbc.8.ip q1, %[pval], 1                         \n"
        "addi  %[pidx], %[pidx], 2                             \n"
        "beqz  %[k], 5f                                        \n"
        "1:                                                    \n"
        "  lhu   %[wrow], 0(%[pidx])                           \n"
        "  slli  %[wrow], %[wrow], 7                           \n"
        "  add   %[wrow], %[wrow], %[wb]                       \n"
        "  esp.vmulas.s8.qacc.ld.xp q2, %[wrow], %[z], q0, q1  \n"
        "  esp.vldbc.8.ip q3, %[pval], 1                       \n"
        "  addi  %[pidx], %[pidx], 2                           \n"
        "  addi  %[k], %[k], -1                                \n"
        "  beqz  %[k], 3f                                      \n"
        "  lhu   %[wrow], 0(%[pidx])                           \n"
        "  slli  %[wrow], %[wrow], 7                           \n"
        "  add   %[wrow], %[wrow], %[wb]                       \n"
        "  esp.vmulas.s8.qacc.ld.xp q0, %[wrow], %[z], q2, q3  \n"
        "  esp.vldbc.8.ip q1, %[pval], 1                       \n"
        "  addi  %[pidx], %[pidx], 2                           \n"
        "  addi  %[k], %[k], -1                                \n"
        "  bnez  %[k], 1b                                      \n"
        "5:                                                    \n"
        "  esp.vmulas.s8.qacc q0, q1                           \n"
        "  j 4f                                                \n"
        "3:                                                    \n"
        "  esp.vmulas.s8.qacc q2, q3                           \n"
        "4:                                                    \n"
        "esp.st.qacc.l.l.128.ip %[out], 16                     \n"
        "esp.st.qacc.l.h.128.ip %[out], 16                     \n"
        "esp.st.qacc.h.l.128.ip %[out], 16                     \n"
        "esp.st.qacc.h.h.128.ip %[out], 16                     \n"
        ".option pop                                           \n"
        : [pidx] "+r"(pidx), [pval] "+r"(pval), [wrow] "=&r"(wrow), [k] "+r"(k), [out] "+r"(out)
        : [wb] "r"(wb), [z] "r"(zero)
        : "memory");
}

void __wrap_rf_axpy_acc16_sp(const uint16_t *idx, const int8_t *val, int m,
                             const int8_t *Wt, int O, const int32_t *b,
                             const int32_t *M, const uint8_t *s, int16_t *res) {
    if (!rf_pie_enabled || O != 128 || m <= 0 || (O & 15) != 0 ||
        ((uintptr_t)Wt & 15u) != 0) {
        __real_rf_axpy_acc16_sp(idx, val, m, Wt, O, b, M, s, res);
        return;
    }
    for (int ob = 0; ob < O; ob += 16) {
        int32_t raw[16] __attribute__((aligned(16)));
        pf_pie_axpy16_raw(idx, val, m, Wt + ob, raw);
        for (int j = 0; j < 16; j++) {
            int o = ob + j;
            int32_t acc = b[o] + raw[s_lane_of[j]];
            res[o] = rf_sat16((int64_t)res[o] + rf_rq(acc, M[o], s[o]));
        }
    }
}

/* ---- 起動時の自己テスト ---------------------------------------------------
 * 参照実装（スカラ）と PIE を同じ乱数入力で比べる。落ちたら rf_pie_enabled = 0 にして
 * すべてスカラに戻す（遅いが正しい）。戻り値 0 = すべて一致。 */
static uint32_t s_rng = 0x12345678u;
static int32_t rnd(void) { s_rng = s_rng * 1664525u + 1013904223u; return (int32_t)(s_rng >> 8); }

int pf_pie_selftest(void) {
    int fails = 0, ufails = 0;
    /* 1. 内積: K ∈ {16, 32, 64, 128, 384, 512}、整列 / 非整列 x、極値（±127 で埋めた K=512） */
    static int8_t w[512 + 16] __attribute__((aligned(16)));
    static int8_t x[512 + 16] __attribute__((aligned(16)));
    const int Ks[] = {16, 32, 64, 128, 384, 512};
    for (int t = 0; t < 6; t++) {
        int K = Ks[t];
        for (int rep = 0; rep < 8; rep++) {
            for (int i = 0; i < K + 16; i++) {
                w[i] = (int8_t)(rep == 6 ? 127 : rep == 7 ? -127 : rnd() % 255 - 127);
                x[i] = (int8_t)(rep == 6 ? 127 : rep == 7 ? 127 : rnd() % 255 - 127);
            }
            for (int off = 0; off < 16; off++) { /* off≠0: x を非整列に */
                const int8_t *xp = x + off;
                int32_t ref = 12345;
                for (int k = 0; k < K; k++) ref += (int32_t)w[k] * xp[k];
                int32_t got = rf_dot_i8(w, xp, K, 12345);
                if (got != ref) {
                    printf("PIE selftest: dot K=%d rep=%d off=%d ref=%ld got=%ld\n", K, rep, off,
                           (long)ref, (long)got);
                    fails++;
                }
                if (off) { /* 非整列の PIE 経路そのものも直接確かめる */
                    int32_t gu = 12345 + rf_pie_dot16u(w, xp, K >> 4);
                    if (gu != ref) ufails++;
                }
            }
        }
    }
    if (ufails == 0) rf_pie_unaligned_ok = 1;
    printf("PIE selftest: unaligned-x PIE path %s\n", ufails ? "MISMATCH -> memcpy fallback" : "OK");
    /* 1b. rf_rq（影のヘッダで mulh 化）を汎用式と照合。s ∈ [1, 58]、acc は int32 / int64 の両方 */
    for (int rep = 0; rep < 4000; rep++) {
        uint8_t sh = (uint8_t)(1 + rnd() % 58);
        int32_t M = (int32_t)(rnd() ^ (rnd() << 12));
        int64_t acc = (rep & 1) ? (int64_t)(int32_t)(rnd() ^ (rnd() << 13))
                                : ((int64_t)(rnd() % 4096 - 2048) << 32) + (int32_t)rnd();
        if (rep % 7 == 0) acc = (rep & 8) ? INT32_MAX : INT32_MIN;
        if (rep % 11 == 0) M = (rep & 16) ? INT32_MAX : INT32_MIN + 1;
        /* |acc·M| < 2^62 の前提（rf_ops.h の注記）を満たす組だけ比べる */
        int64_t p = acc * (int64_t)M;
        if (p > ((int64_t)1 << 61) || p < -((int64_t)1 << 61)) continue;
        int64_t ref = (p + ((int64_t)1 << (sh - 1))) >> sh;
        int64_t got = rf_rq(acc, M, sh);
        if (ref != got) {
            if (fails < 5) printf("PIE selftest: rq acc=%lld M=%ld s=%u ref=%lld got=%lld\n", (long long)acc, (long)M,
                                  sh, (long long)ref, (long long)got);
            fails++;
        }
    }
    /* 1c. rmsnorm（--wrap 版）を参照実装と照合。振幅を 1 〜 32767 で振り、ほぼゼロの入力も含む */
    {
        static int16_t xr[128], Gr[128], Br[128];
        static int8_t yr[128], yw[128];
        for (int rep = 0; rep < 300; rep++) {
            int K = (rep & 1) ? 128 : 32;
            int amp = (rep % 10 == 0) ? 1 : (rep % 10 == 1) ? 3 : (1 << (rep % 15)) + 1;
            for (int k = 0; k < 128; k++) {
                xr[k] = (int16_t)(rnd() % (2 * amp + 1) - amp);
                if (rep % 10 == 2 && k > 2) xr[k] = 0; /* ほぼゼロ → a8 が小さく R ≥ 2^31 */
                Gr[k] = (int16_t)(rnd() % 32767 - 16383);
                Br[k] = (int16_t)(rnd() % 32767 - 16383);
            }
            if (rep % 10 == 3) memset(xr, 0, sizeof xr); /* ss = 0 → a8 = 0 */
            __real_rf_rmsnorm_i16_to_i8(xr, Gr, Br, K, yr);
            __wrap_rf_rmsnorm_i16_to_i8(xr, Gr, Br, K, yw);
            if (memcmp(yr, yw, (size_t)K) != 0) {
                if (fails < 5) printf("PIE selftest: rmsnorm K=%d amp=%d mismatch\n", K, amp);
                fails++;
            }
        }
    }
    /* 1d. isqrt64 / 2^38÷d の厳密性 */
    for (int rep = 0; rep < 3000; rep++) {
        uint64_t v = ((uint64_t)rnd() << 22) ^ (uint64_t)rnd();
        v >>= (rep % 47);
        if (rep % 5 == 0) v = (uint64_t)rnd() * rnd(); /* 完全平方の近く */
        if (rep % 5 == 1) v = (uint64_t)(rnd() & 0x3FFFFF) * (rnd() & 0x3FFFFF);
        if (rf_isqrt64(v) != pf_isqrt64_fast(v)) { if (fails < 5) printf("PIE selftest: isqrt v=%llu\n", (unsigned long long)v); fails++; }
        uint32_t d = (uint32_t)rnd() >> (rep % 24);
        if (!d) d = 1;
        if (rep % 9 == 0) d = 1 + rep / 9;
        if ((((int64_t)1 << 38) / d) != pf_div238(d)) { if (fails < 5) printf("PIE selftest: div238 d=%lu\n", (unsigned long)d); fails++; }
    }
    /* 1e. 列方向の行列積（転置重み）を参照カーネルと照合: K ∈ {32,128}、O ∈ {128,384,512}、3 エピローグ */
    {
        static int8_t Wm[512 * 128] __attribute__((aligned(16)));
        static int8_t WTm[512 * 128 + 16] __attribute__((aligned(16)));
        static int8_t xm[2 * 128] __attribute__((aligned(16)));
        static int8_t ya0[512], ya1[512], yb0[512], yb1[512];
        static int16_t ra0[512], ra1[512], rb0[512], rb1[512];
        static int32_t bm[512], Mm[512];
        static uint8_t sm[512];
        const int Km[] = {32, 128, 128, 128}, Om[] = {128, 384, 512, 128};
        for (int t = 0; t < 4; t++) {
            int K = Km[t], O = Om[t];
            for (int rep = 0; rep < 3; rep++) {
                for (int i = 0; i < K * O; i++) Wm[i] = (int8_t)(rnd() % 255 - 127);
                for (int og = 0; og < O / 16; og++) for (int k = 0; k < K; k++) for (int j = 0; j < 16; j++)
                    WTm[(og * K + k) * 16 + j] = Wm[(og * 16 + j) * K + k];
                for (int i = 0; i < 2 * K; i++) xm[i] = (int8_t)(rnd() % 255 - 127);
                for (int o = 0; o < O; o++) { bm[o] = rnd() % 200000 - 100000; Mm[o] = (1 << 22) + rnd() % 100000; sm[o] = (uint8_t)(34 + rnd() % 8); ra0[o] = rb0[o] = (int16_t)(rnd() % 2000 - 1000); ra1[o] = rb1[o] = (int16_t)(rnd() % 2000 - 1000); }
                __real_rf_linear2_i8(xm, Wm, bm, Mm, sm, K, O, rep & 1, ya0, ya1);
                pf_pie_colmat_test(xm, WTm, bm, Mm, sm, K, O, (rep & 1) ? 1 : 0, yb0, NULL);
                pf_pie_colmat_test(xm + K, WTm, bm, Mm, sm, K, O, (rep & 1) ? 1 : 0, yb1, NULL);
                if (memcmp(ya0, yb0, (size_t)O) || memcmp(ya1, yb1, (size_t)O)) { if (fails < 5) printf("PIE selftest: colmat linear2 K=%d O=%d\n", K, O); fails++; }
                __real_rf_relu2sq2_i8(xm, Wm, bm, Mm, sm, K, O, ya0, ya1);
                pf_pie_colmat_test(xm, WTm, bm, Mm, sm, K, O, 2, yb0, NULL);
                pf_pie_colmat_test(xm + K, WTm, bm, Mm, sm, K, O, 2, yb1, NULL);
                if (memcmp(ya0, yb0, (size_t)O) || memcmp(ya1, yb1, (size_t)O)) { if (fails < 5) printf("PIE selftest: colmat relu2sq2 K=%d O=%d\n", K, O); fails++; }
                __real_rf_linear2_i8_acc16(xm, Wm, bm, Mm, sm, K, O, ra0, ra1);
                pf_pie_colmat_test(xm, WTm, bm, Mm, sm, K, O, 3, NULL, rb0);
                pf_pie_colmat_test(xm + K, WTm, bm, Mm, sm, K, O, 3, NULL, rb1);
                if (memcmp(ra0, rb0, (size_t)O * 2) || memcmp(ra1, rb1, (size_t)O * 2)) { if (fails < 5) printf("PIE selftest: colmat acc16 K=%d O=%d\n", K, O); fails++; }
            }
        }
    }
    /* 2. 疎な axpy（fc2）: レーン順を単位ベクトルで確定してから乱数で照合 */
    static int8_t Wt[512 * 128] __attribute__((aligned(16)));
    static uint16_t idx[512];
    static int8_t val[512];
    static int16_t res_ref[128], res_pie[128];
    static int32_t bb[128], MM[128];
    static uint8_t ss[128];
    /* 2a. レーン順: Wt[0][j] = j+1、val[0] = 1、m = 1 → raw[lane] を読む */
    memset(Wt, 0, sizeof Wt);
    for (int j = 0; j < 16; j++) Wt[j] = (int8_t)(j + 1);
    idx[0] = 0;
    val[0] = 1;
    {
        int32_t raw[16] __attribute__((aligned(16)));
        pf_pie_axpy16_raw(idx, val, 1, Wt, raw);
        int ok = 1;
        uint8_t map[16];
        for (int j = 0; j < 16; j++) {
            int found = -1;
            for (int l = 0; l < 16; l++)
                if (raw[l] == j + 1) found = l;
            if (found < 0) ok = 0;
            map[j] = (uint8_t)(found < 0 ? j : found);
        }
        printf("PIE selftest: QACC raw =");
        for (int l = 0; l < 16; l++) printf(" %ld", (long)raw[l]);
        printf("  (%s)\n", ok ? "lane map resolved" : "UNEXPECTED");
        if (ok) memcpy(s_lane_of, map, 16);
        else fails++;
    }
    /* 2b. 乱数照合 */
    for (int rep = 0; rep < 6; rep++) {
        int m = (rep == 0) ? 1 : (rep == 5 ? 512 : 37 + rep * 90);
        for (size_t i = 0; i < sizeof Wt; i++) Wt[i] = (int8_t)(rnd() % 255 - 127);
        for (int i = 0; i < m; i++) { idx[i] = (uint16_t)(rnd() % 512); val[i] = (int8_t)(rnd() % 255 - 127); }
        for (int o = 0; o < 128; o++) {
            bb[o] = rnd() % 2000 - 1000;
            MM[o] = 1 << 20;   /* rq: (acc·M) >> s。s ≥ 33 の高速経路も通す */
            ss[o] = (uint8_t)(33 + (o & 3));
            res_ref[o] = res_pie[o] = (int16_t)(rnd() % 200 - 100);
        }
        __real_rf_axpy_acc16_sp(idx, val, m, Wt, 128, bb, MM, ss, res_ref);
        __wrap_rf_axpy_acc16_sp(idx, val, m, Wt, 128, bb, MM, ss, res_pie);
        if (memcmp(res_ref, res_pie, sizeof res_ref) != 0) {
            int first = -1;
            for (int o = 0; o < 128; o++) if (res_ref[o] != res_pie[o]) { first = o; break; }
            printf("PIE selftest: axpy m=%d mismatch at o=%d ref=%d got=%d\n", m, first,
                   (int)res_ref[first], (int)res_pie[first]);
            fails++;
        }
    }
    /* 3. VAE の疎な畳み込みの gather（pf_pie_decode.c）: O ∈ {8,16,32,64,128} × 非ゼロ数 */
    for (int t = 0; t < 5; t++) {
        const int O = 8 << t;
        const int C = 64;
        int8_t *W = Wt; /* 16 整列、C·O ≤ 8192 バイト */
        static uint8_t ch[256];
        static int8_t cv[256];
        for (int rep = 0; rep < 4; rep++) {
            int n = (rep == 0) ? 1 : 3 + rep * 40;
            for (int i = 0; i < C * O; i++) W[i] = (int8_t)(rnd() % 255 - 127);
            for (int i = 0; i < n; i++) { ch[i] = (uint8_t)(rnd() % C); cv[i] = (int8_t)(rnd() % 255 - 127); }
            int32_t ref[16] = {0}, got[16] = {0};
            int lanes = O < 16 ? O : 16;
            for (int i = 0; i < n; i++)
                for (int j = 0; j < lanes; j++) ref[j] += (int32_t)cv[i] * W[ch[i] * O + j];
            pf_pie_gather_test(ch, cv, n, W, O, got);
            if (memcmp(ref, got, (size_t)lanes * sizeof(int32_t)) != 0) {
                printf("PIE selftest: gather O=%d n=%d mismatch (ref[0]=%ld got[0]=%ld)\n", O, n,
                       (long)ref[0], (long)got[0]);
                fails++;
            }
        }
    }
    /* 4. マイクロベンチ（正しさとは無関係。最適化の指標）: 16 MAC チャンクあたりの ns */
    {
        int32_t sink = 0;
        int64_t t = esp_timer_get_time();
        for (int i = 0; i < 20000; i++) sink += rf_dot_i8(w, x, 128, i);
        double ns128 = (double)(esp_timer_get_time() - t) * 1000.0 / (20000.0 * 8);
        t = esp_timer_get_time();
        for (int i = 0; i < 5000; i++) sink += rf_dot_i8(w, x, 512, i);
        double ns512 = (double)(esp_timer_get_time() - t) * 1000.0 / (5000.0 * 32);
        static uint8_t gch[64];
        static int8_t gval[64];
        for (int i = 0; i < 64; i++) { gch[i] = (uint8_t)(i * 7 % 64); gval[i] = (int8_t)(i - 32); }
        int32_t gacc[16];
        t = esp_timer_get_time();
        for (int i = 0; i < 5000; i++) { pf_pie_gather_test(gch, gval, 64, Wt, 128, gacc); sink += gacc[0]; }
        double nsg = (double)(esp_timer_get_time() - t) * 1000.0 / (5000.0 * 64);
        printf("PIE bench: dot K=128 %.1f ns/16MAC, K=512 %.1f ns/16MAC, gather O=128 %.1f ns/nonzero (sink %ld)\n",
               ns128, ns512, nsg, (long)sink);
    }
    if (fails) {
        rf_pie_enabled = 0;
        printf("PIE selftest: %d FAIL -> PIE disabled, scalar fallback\n", fails);
    } else {
        printf("PIE selftest: all OK (dot, rq, rmsnorm, isqrt/div, colmat 4x3x3, axpy, gather)\n");
    }
    return fails;
}

/* ---- カーネル別マイクロベンチ（起動時に 1 回。正しさとは無関係）--------------
 * dit.c の各段が 1 トークンあたりに呼ぶ回数と掛けると、段ごとの時間の内訳が分かる:
 *   norm_qkv : rmsnorm128 ×1 + linear2(K128,O384) ×1/2 + rmsnorm32 ×8 + vT 転置
 *   attn     : (dot K32 ×64 + softmax64 + dot K64 非整列 ×32 + rq ×32) × 4 ヘッド
 *   mlp      : rmsnorm128 + relu2sq2(K128,O512) ×1/2 + compact512 + axpy(m≈256)
 *   proj     : linear2_acc16(K128,O128) ×1/2 */
void pf_pie_bench(void) {
    static int8_t W[512 * 128] __attribute__((aligned(16)));
    static int8_t x[2][128] __attribute__((aligned(16)));
    static int8_t y0[512], y1[512];
    static int16_t r16[128], r16b[128];
    static int32_t b[512], M[512], sc[64];
    static uint8_t s[512];
    static int16_t G[128], B[128];
    static uint16_t nzi[512];
    static int8_t nzv[512], prow[64 + 4];
    static uint16_t lut[512];
    int32_t sink = 0;
    for (int i = 0; i < 512 * 128; i++) W[i] = (int8_t)(rnd() % 255 - 127);
    for (int i = 0; i < 256; i++) x[0][i] = (int8_t)(rnd() % 255 - 127);
    for (int i = 0; i < 512; i++) { b[i] = rnd() % 4000 - 2000; M[i] = (1 << 24) + rnd() % 1000; s[i] = (uint8_t)(33 + (i & 3)); }
    for (int i = 0; i < 128; i++) { r16[i] = (int16_t)(rnd() % 4000 - 2000); G[i] = (int16_t)(8000 + rnd() % 8000); B[i] = (int16_t)(rnd() % 200 - 100); }
    for (int i = 0; i < 64; i++) sc[i] = rnd() % 100000;
    for (int i = 0; i < 512; i++) lut[i] = (uint16_t)(65535 >> (i / 40));
    for (int i = 0; i < 68; i++) prow[i] = (int8_t)(rnd() % 127);
    int64_t t;
#define BENCH(label, reps, div, stmt)                                                     \
    t = esp_timer_get_time();                                                             \
    for (int it = 0; it < (reps); it++) { stmt; }                                         \
    printf("PIE bench2: %-28s %8.1f ns\n", label, (double)(esp_timer_get_time() - t) * 1000.0 / ((double)(reps) * (div)));
    BENCH("rmsnorm K=128 (ref)", 2000, 1, __real_rf_rmsnorm_i16_to_i8(r16, G, B, 128, y0); sink += y0[it & 127]);
    BENCH("rmsnorm K=128 (wrap)", 2000, 1, __wrap_rf_rmsnorm_i16_to_i8(r16, G, B, 128, y0); sink += y0[it & 127]);
    BENCH("rmsnorm K=32 (wrap)", 4000, 1, __wrap_rf_rmsnorm_i16_to_i8(r16, G, B, 32, y0); sink += y0[it & 31]);
    BENCH("linear2 K=128 O=384 (2 tok)", 200, 1, rf_linear2_i8(x[0], W, b, M, s, 128, 384, 0, y0, y1); sink += y0[it & 127]);
    BENCH("colmat  K=128 O=384 (2 tok)", 200, 1, pf_pie_colmat_test(x[0], W, b, M, s, 128, 384, 0, y0, NULL); pf_pie_colmat_test(x[1], W, b, M, s, 128, 384, 0, y1, NULL); sink += y0[it & 127]);
    BENCH("colmat sq K=128 O=512 (2 tok)", 200, 1, pf_pie_colmat_test(x[0], W, b, M, s, 128, 512, 2, y0, NULL); pf_pie_colmat_test(x[1], W, b, M, s, 128, 512, 2, y1, NULL); sink += y0[it & 127]);
    BENCH("linear2_acc16 K=128 O=128", 500, 1, rf_linear2_i8_acc16(x[0], W, b, M, s, 128, 128, r16, r16b); sink += r16[it & 127]);
    BENCH("relu2sq2 K=128 O=512 (2 tok)", 200, 1, rf_relu2sq2_i8(x[0], W, b, M, s, 128, 512, y0, y1); sink += y1[it & 127]);
    BENCH("softmax N=64", 2000, 1, rf_softmax_i32_to_i8(sc, 64, 1 << 20, 30, lut, 511, prow); sink += prow[it & 63]);
    BENCH("compact n=512", 2000, 1, sink += rf_compact_i8(y0, 512, nzi, nzv));
    int m = 0;
    for (int i = 0; i < 512; i += 2) { nzi[m] = (uint16_t)i; nzv[m] = (int8_t)(rnd() % 255 - 127); m++; }
    BENCH("axpy_sp m=256 O=128 (PIE)", 500, 1, __wrap_rf_axpy_acc16_sp(nzi, nzv, m, W, 128, b, M, s, r16); sink += r16[it & 127]);
    BENCH("axpy_sp m=256 O=128 (ref)", 200, 1, __real_rf_axpy_acc16_sp(nzi, nzv, m, W, 128, b, M, s, r16); sink += r16[it & 127]);
    BENCH("requant n=32", 4000, 1, rf_requant_i16_to_i8(r16, 32, 1 << 20, 30, y0); sink += y0[it & 31]);
    BENCH("dot K=32 aligned", 20000, 1, sink += rf_dot_i8(W + (it & 63) * 32, x[0], 32, 0));
    BENCH("dot K=64 x unaligned", 20000, 1, sink += rf_dot_i8(W + (it & 63) * 64, prow + 1, 64, 0));
    BENCH("rq (int64 mul+shift)", 20000, 1, sink += (int32_t)rf_rq(sink + it, M[it & 511], s[it & 511]));
#undef BENCH
    printf("PIE bench2: (sink %ld)\n", (long)sink);
}

/* ---- PIE 命令そのもののスループット（メモリの影響を除いた下限を知る）------------------
 * 各ループは同じ命令を 16 個並べて 1000 回回す。ns/命令で表示 */
#define PIE_LOOP(label, body)                                                                          \
    do {                                                                                               \
        int64_t t0 = esp_timer_get_time(); /* ⚠️ レジスタ固定変数の初期化より前に呼ぶ（a2〜a5 は呼び出しで壊れる） */ \
        register int k __asm__("a4") = 1000;                                                           \
        register const int8_t *p __asm__("a2") = buf;                                                  \
        register const int8_t *q __asm__("a3") = buf + 64;                                             \
        register int st __asm__("a5") = 16;                                                            \
        __asm__ volatile(".option push\n.option arch, +xesppie\n"                                      \
                         "esp.zero.qacc\nesp.zero.xacc\n"                                              \
                         "esp.vld.128.ip q0, %[p], 0\nesp.vld.128.ip q1, %[q], 0\n"                    \
                         "1:\n" body body body body body body body body                                \
                         body body body body body body body body                                       \
                         "addi %[k], %[k], -1\nbnez %[k], 1b\n.option pop\n"                           \
                         : [k] "+r"(k), [p] "+r"(p), [q] "+r"(q) : [st] "r"(st) : "memory");           \
        int64_t t1 = esp_timer_get_time();                                                             \
        printf("PIE bench3: %-34s %6.2f ns/insn\n", label, (double)(t1 - t0) * 1000.0 / 16000.0);       \
    } while (0)

void pf_pie_bench3(void) {
    static int8_t buf[4096] __attribute__((aligned(16)));
    PIE_LOOP("vmulas.s8.qacc q0,q1 (dep chain)", "esp.vmulas.s8.qacc q0, q1\n");
    PIE_LOOP("vmulas.s8.xacc q0,q1 (dep chain)", "esp.vmulas.s8.xacc q0, q1\n");
    PIE_LOOP("vld.128.ip (L1, no advance)", "esp.vld.128.ip q2, %[p], 0\n");
    PIE_LOOP("vldbc.8.ip (no advance)", "esp.vldbc.8.ip q3, %[q], 0\n");
    PIE_LOOP("vmulas.s8.qacc.ld.xp fused", "esp.vmulas.s8.qacc.ld.xp q2, %[p], %[st], q0, q1\naddi %[p], %[p], -16\n");
    PIE_LOOP("vsmulas.s8.qacc q0,q1,3", "esp.vsmulas.s8.qacc q0, q1, 3\n");
    PIE_LOOP("vsmulas.s8.qacc.ld.incp fused", "esp.vsmulas.s8.qacc.ld.incp q2, %[p], q0, q1, 3\naddi %[p], %[p], -16\n");
    PIE_LOOP("vmulas.qacc + vld interleaved", "esp.vmulas.s8.qacc q0, q1\nesp.vld.128.ip q2, %[p], 0\n");
    PIE_LOOP("vmulas.qacc + vldbc interleaved", "esp.vmulas.s8.qacc q0, q1\nesp.vldbc.8.ip q3, %[q], 0\n");
    PIE_LOOP("addi only (scalar)", "addi %[st], %[st], 0\n");
}
