/* ESP32-P4 PIE（RISC-V 拡張 xesppie）による int8 内積カーネル。
 *
 * 上流 rf_ops.h の `#else`（参照実装）の手前に、影のヘッダ（make_shadow.py が生成）が
 *   #elif defined(RF_PIE_P4)
 *   #include "rf_ops_pie.h"
 * を差し込んで取り込む。ここで定義する 4 関数は参照実装と同じ署名・同じ結果（bit 一致）。
 *
 * 契約（rf_ops.h の "Width contract" と同じ）:
 *   rf_dot_i8(w, x, K, acc) = acc + Σ w[k]·x[k]（int32 の二の補数加算。モデル設計で |Σ| < 2^31）
 *   PIE の XACC は 40 bit なので途中で溢れず、下位 32 bit を取り出せば int32 の結果に等しい。
 *   加算の順序は結果に影響しない（int32 加算は順序独立）。
 *
 * 使う命令（esp-dl の esp32p4-pie-simd スキルの資料に基づく）:
 *   esp.zero.xacc                 XACC = 0
 *   esp.vld.128.ip q, rs, 16      128 bit ロード + rs += 16   ⚠️ rs は 16 バイト境界必須
 *   esp.vmulas.s8.xacc qx, qy     16 レーンの s8×s8 の和を XACC に加算
 *   esp.movx.r.xacc.l rd          XACC[23:0]（下位 24 bit。符号拡張）
 *   esp.movx.r.xacc.h rd          XACC[39:24]（上位 16 bit）
 *   ⚠️ .l は 32 bit ではなく **24 bit** しか返さない。このモデルの和は最大でも
 *      512·127² ≈ 8.26M < 2^23 なので .l だけでも正しいが、余裕が 1.6% しかないので
 *      .h と結合して下位 32 bit を組み立てる（|Σ| < 2^31 なら二の補数の下位 32 bit = 結果）。
 * ⚠️ PIE 命令のオペランドは x8–x15 / x24–x31 しか取れない。GCC の "r" は x16–x23 も割り当てる
 *    ので、レジスタを a2〜a5 に固定した変数で渡す。
 * ⚠️ IDF の -march に xesppie が無いので `.option arch, +xesppie` を asm の中で足す。
 *
 * 整列: 影のヘッダで RF_ALIGN4 を aligned(16) にしたので、エンジンの静的バッファと局所配列
 * （arena、res/xa/qkv/vT、x_in/xm/xf/h1）は 16 整列になり、行ストライド K ∈ {16,32,64,128,384,512}
 * は 16 の倍数。したがって DiT の密な行列積と attention の scores は整列済み。
 * 整列していない x（attention の prow など属性の無い局所配列）は 16 整列の一時領域へ複写してから
 * 使う。整列していない w、または K が 16 の倍数でない呼び出し（VAE の C=8 の畳み込み）は
 * スカラの参照ループに落とす。どの経路でも結果は同じ。
 *
 * rf_pie_enabled: 起動時の自己テスト（pf_pie_selftest）が参照実装との不一致を見つけたら 0 にする。
 * 0 のときはすべてスカラに落ちる（遅いが正しい）。 */
#ifndef RF_OPS_PIE_H
#define RF_OPS_PIE_H

#include <stdint.h>
#include <string.h>

#define RF_PIE_XTMP 512 /* 一時複写する x の最大長（K の最大は 4·RF_DIM = 512） */

extern int rf_pie_enabled;

static inline int rf_pie_al16(const void *p) { return ((uintptr_t)p & 15u) == 0; }

/* 両ポインタ 16 整列、k16 = K/16 ≥ 1。
 * ソフトウェアパイプライン: ロード直後に MAC で使うと停止するので、融合命令
 *   esp.vmulas.s8.xacc.ld.ip qu, rs, 16, qx, qy   （XACC += qx·qy、続けて qu ← [rs]、rs += 16）
 * で「今の MAC」と「次のロード」を 1 命令にし、(q0,q1) と (q2,q3) を交互に使う。
 * 先頭でチャンク 0 を読み、ループ 1 周で 2 チャンク分の MAC と 2 チャンク先読み。
 * 残り (k16−1) が奇数なら最後に 1 チャンクぶん追加。合計 MAC 回数 = k16、ロード = k16。 */
static inline int32_t rf_pie_dot16(const int8_t *a, const int8_t *b, int k16) {
    register const int8_t *pa __asm__("a2") = a;
    register const int8_t *pb __asm__("a3") = b;
    register int pairs __asm__("a4") = (k16 - 1) >> 1;
    register int odd __asm__("a5") = (k16 - 1) & 1;
    register int32_t lo __asm__("a0");
    register int32_t hi __asm__("a1");
    __asm__ volatile(
        ".option push                                       \n"
        ".option arch, +xesppie                             \n"
        "esp.zero.xacc                                      \n"
        "esp.vld.128.ip q0, %[pa], 16                       \n"
        "esp.vld.128.ip q1, %[pb], 16                       \n"
        "beqz %[pairs], 2f                                  \n"
        "1:                                                 \n"
        "  esp.vmulas.s8.xacc.ld.ip q2, %[pa], 16, q0, q1   \n"
        "  esp.vld.128.ip q3, %[pb], 16                     \n"
        "  esp.vmulas.s8.xacc.ld.ip q0, %[pa], 16, q2, q3   \n"
        "  esp.vld.128.ip q1, %[pb], 16                     \n"
        "  addi %[pairs], %[pairs], -1                      \n"
        "  bnez %[pairs], 1b                                \n"
        "2:                                                 \n"
        "beqz %[odd], 3f                                    \n"
        "  esp.vmulas.s8.xacc.ld.ip q2, %[pa], 16, q0, q1   \n"
        "  esp.vld.128.ip q3, %[pb], 16                     \n"
        "  esp.vmulas.s8.xacc q2, q3                        \n"
        "  j 4f                                             \n"
        "3:                                                 \n"
        "  esp.vmulas.s8.xacc q0, q1                        \n"
        "4:                                                 \n"
        "esp.movx.r.xacc.l %[lo]                            \n"
        "esp.movx.r.xacc.h %[hi]                            \n"
        ".option pop                                        \n"
        : [lo] "=r"(lo), [hi] "=r"(hi), [pa] "+r"(pa), [pb] "+r"(pb), [pairs] "+r"(pairs)
        : [odd] "r"(odd)
        : "memory");
    /* XACC[39:0] = hi[15:0]:lo[23:0]。欲しいのは下位 32 bit = hi[7:0]:lo[23:0] */
    return (int32_t)(((uint32_t)hi << 24) | ((uint32_t)lo & 0x00FFFFFFu));
}

/* x が 16 整列でないとき（attention の p·v の prow など）。w は 16 整列、k16 ≥ 1。
 *   esp.ld.128.usar.ip q, rs, 16   rs を切り下げた境界から 16 B 読み、SAR_BYTE = rs & 15、rs += 16
 *   esp.src.q.qup qz, qw, qy       qz = (qy‖qw) >> SAR_BYTE·8（= rs から始まる 16 B）、qw ← qy
 * 連続する 2 ブロックを継ぎ合わせて非整列の 16 B を作る。
 * ⚠️ 最後のブロックは x+K の先まで（最大 15 B）読む。呼び出し側の配列はスタックか静的領域
 *    なので読めるが、メモリ領域の末尾ぴったりに置かれた配列には使えない。 */
static inline int32_t rf_pie_dot16u(const int8_t *a, const int8_t *b, int k16) {
    register const int8_t *pa __asm__("a2") = a;
    register const int8_t *pb __asm__("a3") = b;
    register int k __asm__("a4") = k16;
    register int32_t lo __asm__("a5");
    register int32_t hi __asm__("a0");
    __asm__ volatile(
        ".option push                        \n"
        ".option arch, +xesppie              \n"
        "esp.zero.xacc                       \n"
        "esp.ld.128.usar.ip q4, %[pb], 16    \n"
        "1:                                  \n"
        "  esp.ld.128.usar.ip q5, %[pb], 16  \n"
        "  esp.src.q.qup q1, q4, q5          \n"
        "  esp.vld.128.ip q0, %[pa], 16      \n"
        "  esp.vmulas.s8.xacc q0, q1         \n"
        "  addi %[k], %[k], -1               \n"
        "  bnez %[k], 1b                     \n"
        "esp.movx.r.xacc.l %[lo]             \n"
        "esp.movx.r.xacc.h %[hi]             \n"
        ".option pop                         \n"
        : [lo] "=r"(lo), [hi] "=r"(hi), [pa] "+r"(pa), [pb] "+r"(pb), [k] "+r"(k)
        :
        : "memory");
    return (int32_t)(((uint32_t)hi << 24) | ((uint32_t)lo & 0x00FFFFFFu));
}

extern int rf_pie_unaligned_ok; /* 自己テストで非整列ロードの意味を確認できたら 1 */

static inline int32_t rf_dot_i8(const int8_t *w, const int8_t *x, int K,
                                int32_t acc) {
    if (rf_pie_enabled && K >= 16 && (K & 15) == 0 && rf_pie_al16(w)) {
        if (rf_pie_al16(x)) return acc + rf_pie_dot16(w, x, K >> 4);
        if (rf_pie_unaligned_ok) return acc + rf_pie_dot16u(w, x, K >> 4);
        if (K <= RF_PIE_XTMP) {
            int8_t tmp[RF_PIE_XTMP] __attribute__((aligned(16)));
            memcpy(tmp, x, (size_t)K);
            return acc + rf_pie_dot16(w, tmp, K >> 4);
        }
    }
    for (int k = 0; k < K; k++) acc += (int32_t)w[k] * x[k];
    return acc;
}

static inline void rf_dot2_i8(const int8_t *w0, const int8_t *w1,
                              const int8_t *x, int K, int32_t *a0,
                              int32_t *a1) {
    *a0 = rf_dot_i8(w0, x, K, *a0);
    *a1 = rf_dot_i8(w1, x, K, *a1);
}

/* a[0..7] += v · w[0..7]（疎な経路）。8 レーンの 1 回きりでは PIE の利点が無いのでスカラ。
 * fc2 の疎な経路は rf_axpy_acc16_sp を丸ごと置き換えて QACC で累算する（pf_pie_kernels.c）。 */
static inline void rf_axpy8_i8(const int8_t *w, int32_t v, int32_t *a) {
    for (int j = 0; j < 8; j++) a[j] += v * (int32_t)w[j];
}

/* 2 行 × 2 トークン。x の 2 トークン目は x+K、w の 2 行目は w+K（呼び出し側の規約） */
static inline void rf_dot2x2_i8(const int8_t *w, const int8_t *x, int K,
                                int32_t *a00, int32_t *a01, int32_t *a10,
                                int32_t *a11) {
    *a00 = rf_dot_i8(w, x, K, *a00);
    *a01 = rf_dot_i8(w, x + K, K, *a01);
    *a10 = rf_dot_i8(w + K, x, K, *a10);
    *a11 = rf_dot_i8(w + K, x + K, K, *a11);
}

#endif /* RF_OPS_PIE_H */
