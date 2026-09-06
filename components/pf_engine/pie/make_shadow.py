"""上流の engine/include/rf_ops.h から、PIE 分岐を足した「影のヘッダ」を生成する。

    uv run --no-project python make_shadow.py <upstream rf_ops.h> <out dir>

CMake の configure 時に走る。生成物 <out dir>/rf_ops.h をインクルードパスの**先頭**に置くと、
エンジンの `#include "rf_ops.h"` がこちらを拾う（上流のソースは無改変のまま）。

変更点は 3 つだけ。どちらも文字列の完全一致で当て、当たらなければ止まる
（上流が変わったときに黙って古い経路に落ちないようにするため）:
  1. RF_ALIGN4 を aligned(16) に（PIE の 128 bit ロードは 16 バイト境界が要る）
  2. 参照実装の `#else` の手前に `#elif defined(RF_PIE_P4)` / `#include "rf_ops_pie.h"` を挿入
  3. rf_rq の 64 bit 乗算を、acc が int32 に収まるときは mulh 1 命令にする（結果は同じ）
"""
import pathlib
import sys

src = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
out_dir = pathlib.Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)

A1 = "#define RF_ALIGN4 __attribute__((aligned(4)))"
A2 = ("#else\n"
      "static inline int32_t rf_dot_i8(const int8_t *w, const int8_t *x, int K,\n"
      "                                int32_t acc) {\n"
      "    for (int k = 0; k < K; k++) acc += (int32_t)w[k] * x[k];")
A3 = ("static inline int64_t rf_rq(int64_t acc, int32_t M, uint8_t s) {\n"
      "    int64_t p = acc * (int64_t)M;\n"
      "    if (__builtin_expect(s >= 33, 1)) {\n"
      "        int32_t hi = (int32_t)(p >> 32);\n"
      "        return (hi + (1 << (s - 33))) >> (s - 32);\n"
      "    }\n"
      "    return (p + ((int64_t)1 << (s - 1))) >> s;\n"
      "}")
A3_NEW = ("static inline int64_t rf_rq(int64_t acc, int32_t M, uint8_t s) {\n"
      "    /* PIE 影のヘッダ: 呼び出し側の acc はほぼ常に int32（int64 に広げただけ）。\n"
      "     * その場合 p>>32 は 32×32→64 の上位語 = RISC-V の mulh 1 命令で済む。\n"
      "     * インライン後は `acc == (int32_t)acc` が定数畳み込みされ、分岐は消える。\n"
      "     * 結果は元の式と bit 一致（同じ p の上位 32 bit）。 */\n"
      "    if (__builtin_expect(s >= 33, 1)) {\n"
      "        int32_t hi;\n"
      "        if (__builtin_expect(acc == (int64_t)(int32_t)acc, 1))\n"
      "            hi = (int32_t)(((int64_t)(int32_t)acc * (int64_t)M) >> 32);\n"
      "        else\n"
      "            hi = (int32_t)((acc * (int64_t)M) >> 32);\n"
      "        return (hi + (1 << (s - 33))) >> (s - 32);\n"
      "    }\n"
      "    return (acc * (int64_t)M + ((int64_t)1 << (s - 1))) >> s;\n"
      "}")
for a in (A1, A2, A3):
    if src.count(a) != 1:
        sys.exit(f"make_shadow.py: anchor not found exactly once in {sys.argv[1]}:\n{a}")

hdr = ("/* ⚠️ 生成物。編集しないこと。components/pf_engine/pie/make_shadow.py が\n"
       " * 上流の engine/include/rf_ops.h に PIE 分岐を足したもの。差分は 3 箇所（RF_ALIGN4、\n"
       " * #elif defined(RF_PIE_P4)、rf_rq の mulh 化）。 */\n")
dst = src.replace(A1, "#define RF_ALIGN4 __attribute__((aligned(16))) /* PIE: 128 bit ロードの整列 */")
dst = dst.replace(A2, "#elif defined(RF_PIE_P4)\n#include \"rf_ops_pie.h\"\n" + A2)
dst = dst.replace(A3, A3_NEW)
(out_dir / "rf_ops.h").write_text(hdr + dst, encoding="utf-8")
print(f"make_shadow.py: wrote {out_dir / 'rf_ops.h'}")
