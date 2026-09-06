# pico-faces on M5Stack Tab5

[cpldcpu/pico-faces](https://github.com/cpldcpu/pico-faces)（RP2350 で動く、128×128 の顔画像を生成する
潜在拡散トランスフォーマー。DiT 2.4M + VAE デコーダ、int8 の C99 推論エンジン）を
**M5Stack Tab5（ESP32-P4）** に移植したものです。生成した顔を Tab5 の 1280×720 画面に 5 倍で表示します。

- 推論エンジン（`upstream/engine`）は**無改変**。RP2350 版と同じソースを ESP-IDF でコンパイルします。
- 板固有の部分（画面、タッチ、USB シリアル、2 コア並列）は `main/` に、
  ESP32-P4 の SIMD（PIE）による高速化は `components/pf_engine/pie/` にあります。
- 生成結果は seed ごとに bit 単位で決定的なので、ホスト版・上流の Pico 版と同じ CRC32 で照合できます。
  PIE 版も参照実装と **bit 一致**します（起動時の自己テストと golden CRC で確認）。

| 設定 | 参照 C（移植直後） | PIE 版（現在） |
|---|---:|---:|
| K=4, CFG w=4（golden 規約） | 10.10 s | **2.01 s** |
| K=8, CFG なし | 10.08 s | 約 2.0 s |
| K=8, CFG w=6 | 18.36 s | **3.51 s** |

上流の RP2350 @300 MHz は K=4 w=4 で約 10 秒です。

## 構成

```
CMakeLists.txt              PF_MODEL の選択、モデル blob の埋め込み
main/main.cpp               画面（M5GFX）・タッチ・USB シリアル・生成タスク・起動時の自己テスト呼び出し
main/pf_par.c               rf_par_for を FreeRTOS の 2 タスク（コア 0 / 1）で実装。段ごとのプロファイル
components/pf_engine/       upstream/engine/src/*.c をそのままコンパイルするコンポーネント
components/pf_engine/pie/   ESP32-P4 PIE 版カーネル（下記）
tools/serial_cmd.py         USB シリアルにコマンドを送る補助（DTR/RTS を動かさない）
upstream/                   git submodule: cpldcpu/pico-faces（エンジン、モデル blob、golden）
sdkconfig.defaults          Tab5 の板設定（P4 v1.0、HEX PSRAM、USB Serial/JTAG、L2 キャッシュ 256 KB）
partitions.csv              app 7 MB（blob 4 MB を .rodata に含む）
idf.sh                      ESP-IDF v5.5 をクリーンな環境で有効にして idf.py を呼ぶ
```

## ビルドと書き込み

```bash
git submodule update --init         # 上流を取る（約 300 MB。モデルの重みと golden を含む）
./idf.sh build                      # 既定: m3_decD_deep_full（高品質、blob 4.02 MB）、PIE 有効
./idf.sh -DPF_PIE=0 build           # 参照実装（陰性対照・bit 一致の確認用）
./idf.sh -DPF_MODEL=m3_long_cfg build   # 高速モデル（blob 2.57 MB）。切り替え時は build/ を消す
./idf.sh -p /dev/ttyACM0 flash
```

必要なもの: ESP-IDF v5.5（`~/esp/esp-idf`）、`uv`（影のヘッダの生成に使う）。
M5Unified / M5GFX は component manager が取ってきます。

## 使い方

起動すると自己テスト（約 0.1 秒）のあと、seed 3、8 ステップ、クラス 3（male / smile）、CFG w=6 で 1 枚生成します。

| 操作 | 動作 |
|---|---|
| 画像をタップ | seed を +1 して生成 |
| 右パネルの上半分をタップ | クラスを切り替え（female/no-smile → female/smile → male/no-smile → male/smile → null） |
| 右パネルの下半分をタップ | CFG を切り替え（none → 4 → 6 → 8 → none） |

USB シリアル（115200）からは上流の Pico 版と同じ書式で指示できます。

```
G <seed> [k_steps] [class] [w]     生成。class / w を省略すると上流の golden 規約
                                   （class = seed % 5、w_idx = seed % 4 − 1）
I                                  モデル情報
```

応答は 1 行のテキストで、そのあとに段ごとの所要時間（プロファイル）がログとして続きます。

```
OK seed=1 k=4 class=1 w=4 crc32=40c5e5a0 ms=2011
I (...) pico_faces:   DiT 1513 ms, VAE decode 498 ms (PIE)
  phase              calls      ms
  fn@4000b194       96    408.5      ← nm build/pico_faces_tab5.elf | grep _range で関数名を引く
  ...
  stage(memcpy)     386    147.7  (18440 KB)
```

⚠️ `cat /dev/ttyACM0` や `printf > /dev/ttyACM0` のようにポートを開閉すると、Linux の cdc-acm ドライバが
DTR/RTS を動かし、ESP32-P4 がダウンロードモードに落ちることがあります（実際に落ちました）。
`tools/serial_cmd.py` は pyserial で開く前に DTR/RTS を下げるので安全です。

```bash
uv run --no-project --with pyserial python tools/serial_cmd.py --boot 14 --wait 10 "G 1 4" "G 2 4" "G 3 4"
```

## bit 一致の確認（golden）

上流の golden は `k_steps=4`、class / w は規約どおりで作られています。ホストで同じエンジンを動かした CRC32 は
次の通りで、Tab5 で同じ値が出れば移植は bit 単位で正しいと言えます（PIE 版で 5 通りすべて一致を確認済み）。

| コマンド | crc32 |
|---|---|
| `G 1 4` | `40c5e5a0` |
| `G 2 4` | `630a7574` |
| `G 3 4` | `c76b6c2e` |
| `G 3 8 3 6` | `ecb7bc36` |
| `G 3 8 4 6` | `fd41f027` |

ホスト側の再現:

```bash
gcc -O2 -Iupstream/checkpoints/m3_decD_deep_full -Iupstream/engine/include \
    upstream/engine/desktop/main_golden.c upstream/engine/src/*.c -o rf_golden
./rf_golden upstream/checkpoints/m3_decD_deep_full/model.bin out 4 1 2 3
cmp out/eng_1.rgb upstream/checkpoints/m3_decD_deep_full/goldens/golden_1.rgb
```

## PIE（ESP32-P4 の SIMD）版カーネル

上流のソースを 1 バイトも変えずに、2 つの仕組みで差し替えています。

1. **影のヘッダ**: `pie/make_shadow.py` が configure 時に上流の `rf_ops.h` から `build/pf_pie/rf_ops.h` を生成し、
   インクルードパスの先頭に置きます。差分は 3 箇所だけ（`RF_ALIGN4` を 16 整列に、参照実装の手前に
   `#elif defined(RF_PIE_P4)` で `rf_ops_pie.h` を挿入、`rf_rq` の 64 bit 乗算を `mulh` 1 命令に）。
   上流のヘッダが変わって当たらなくなると configure が止まります。
2. **リンカの `--wrap`**: 別の翻訳単位から呼ばれるカーネル（`rf_axpy_acc16_sp`、`rf_decode`、
   `rf_rmsnorm_i16_to_i8`、`rf_linear*` / `rf_relu2sq*`）を `__wrap_` 版に差し替え、参照実装は `__real_` として
   フォールバックと自己テストに残します。

| ファイル | 内容 |
|---|---|
| `pie/rf_ops_pie.h` | `rf_dot_i8` 系。XACC 累算（`esp.vmulas.s8.xacc`）、融合命令のソフトウェアパイプライン、非整列 x（`esp.ld.128.usar` + `esp.src.q`） |
| `pie/pf_pie_linear.c` | 密な行列積（qkv / proj / fc1 / emb）。重みを**ブロック転置** `[O/16][K][16]` で内部 SRAM にステージし、`esp.vsmulas.s8.qacc.ld.incp` で 16 出力を QACC に同時累算 |
| `pie/pf_pie_kernels.c` | 疎な fc2（QACC gather）、rmsnorm（整数的に厳密な高速版）、起動時の自己テストとマイクロベンチ |
| `pie/pf_pie_decode.c` | VAE デコーダ。疎な 3×3 畳み込みを QACC gather で（O=8 の層は 64 bit ロード） |

**自己テスト**（起動時、約 0.1 秒）: 内積（K 6 種 × 非整列 16 通り）、`rf_rq`（4,000 組）、rmsnorm（300 組）、
isqrt / 除算（3,000 組）、列方向行列積（4 形状 × 3 エピローグ）、fc2 gather、VAE gather を参照実装と突き合わせ、
1 つでも食い違えば `rf_pie_enabled = 0` にしてすべてスカラに戻します（遅いが正しい）。

**実機で分かった PIE の性質**（400 MHz）:

| 項目 | 実測 |
|---|---|
| PIE 命令 1 個（MAC でもロードでも） | 約 3.3 ns ≈ 1.3 サイクル。ロードと MAC は重ならない |
| MAC + ロードの融合命令 | 約 6.1 ns（= 2 命令ぶん。命令数は減るが時間は減らない） |
| 16 MAC あたりの下限 | 約 6.2 ns（重みロード 1 + MAC 1） |
| `esp.movx.r.xacc.l` | 下位 **24 bit** しか返さない（`.h` と結合して 32 bit を組み立てる） |
| QACC のレーン順（s8 MAC） | `st.qacc.l.l / l.h / h.l / h.h` の順で int32[16] の恒等 |
| PSRAM 上の重み | L1 ミス → L2 のレイテンシで PIE のロードが止まり、1 行 19 ns。内部 SRAM に置くと 2〜3 倍速い |

## 設計メモ

- **重みの置き場所**: モデル blob は app の `.rodata` に埋め込み、起動時に PSRAM（32 MB）へコピーします。
  密な層の転置コピー（約 1.5 MB）も PSRAM に作ります。DiT の各層はそこから**内部 SRAM の 200 KB**
  （`pf_pie_linear.c` の staging buffer。上流の arena スロットと同じ相対配置）へ memcpy してから使い、
  VAE の疎な重み（最大 147 KB）も同じ領域を時分割で使います。
- **2 コア**: 上流の `firmware/par.c`（core1 を FIFO で起こす）と同じ契約を FreeRTOS の task notification で
  実装しました。生成タスクはコア 0、ワーカーはコア 1 に固定します（VAE デコーダのスクラッチが
  `rf_core_id()` で分かれているため）。IDF v5.5 の FreeRTOS は文脈切替で PIE のレジスタを退避します。
- **内部 RAM**: エンジンの静的領域（arena 256 KB + DiT の活性 88 KB）と PIE の自己テスト用バッファは
  `components/pf_engine/linker.lf` で PSRAM（`.ext_ram.bss`）に置いています。
  ⚠️ IDF 付属の `extram_bss` スキームは esp_wifi の fragment にあり、Wi-Fi を含まないこのビルドでは
  黙って無視されます。自前のスキーム（`bss -> extern_ram`）を定義する必要がありました。
- **レジスタ固定変数**: PIE 命令のオペランドは x8–x15 / x24–x31 しか取れないので `register ... asm("a2")` で
  渡していますが、初期化から asm までの間に関数呼び出しを挟むと壊れます（起動時に Load access fault で
  再起動を繰り返す事故を 1 回起こしました）。

## 残っている高速化の候補（K=4 w=4 の 2.01 s の内訳）

| 段 | ms | 手 |
|---|---:|---|
| mlp（fc1 + 疎 fc2） | 600 | fc2 の gather を L1 に収まる単位に分けて QACC を退避／復帰 |
| norm_qkv | 409 | rmsnorm の固定コスト（isqrt・除算）をさらに削る |
| VAE 疎な畳み込み | 390 | 同上の分割 |
| attention | 256 | softmax の 64 回の除算を逆数乗算（厳密）に |
| 重みステージング memcpy | 148 | `esp_async_memcpy`（GDMA）で計算と重ねる |
| VAE 密な畳み込み（C=8） | 104 | スカラのまま。im2col + PIE 内積 |
| proj | 72 | — |

## 状態

- [x] ホストで上流エンジンをビルドし、golden 3 枚と byte 一致
- [x] 実機で生成・表示・CRC 照合（2026-09-06）
- [x] PIE 版（内積 → 疎 fc2 → VAE → 列方向行列積 → 内部 SRAM ステージング）。5 通りの golden と bit 一致、
      K=4 w=4 で 10.10 s → 2.01 s
- [ ] 起動後しばらく、触っていないのにタッチの「クリック」が来て勝手に生成が始まることがあります
      （起動 3 秒間は無視していますが、その後にも数秒おきに来た回がありました）。座標をログに出すようにしてあります

## ライセンス

上流の cpldcpu/pico-faces にはライセンス表記がありません（2026-09-06 時点）。この移植の `main/` と
`components/` は上流に依存しており、再配布や公開の可否は上流の作者の意向によります。
