# pico-faces on M5Stack Tab5

[cpldcpu/pico-faces](https://github.com/cpldcpu/pico-faces) を **M5Stack Tab5（ESP32-P4）** に移植したものです。
pico-faces は RP2350 で動く 128×128 の顔画像生成モデル（潜在拡散トランスフォーマー DiT 2.4M + VAE デコーダ、
int8 の C99 推論エンジン）です。生成した顔を Tab5 の 1280×720 画面に 5 倍で表示します。

- 推論エンジン（`upstream/engine`）は**無改変**です。板固有の部分と ESP32-P4 の SIMD（PIE）による高速化だけを足しました。
- 生成結果は seed ごとに bit 単位で決定的です。ホスト版・上流の Pico 版・この PIE 版はすべて同じ CRC32 になります。

| 設定 | 参照 C（移植直後） | PIE 版（現在） |
|---|---:|---:|
| K=4、CFG w=4（golden の規約） | 10.10 s | **2.02 s** |
| K=8、CFG なし | 10.08 s | **2.03 s** |
| K=8、CFG w=6 | 18.36 s | **3.53 s** |

上流の RP2350 @300 MHz は K=4 w=4 で約 10 秒です。

## 必要なもの

- M5Stack Tab5（ESP32-P4、PSRAM 32 MB、USB Serial/JTAG）
- ESP-IDF v5.5（`~/esp/esp-idf` に展開してあるものを `idf.sh` が使います）
- `uv`（ビルド中に Python スクリプトを 1 本走らせます）
- M5Unified / M5GFX は component manager が自動で取ってきます

## ビルドと書き込み

```bash
git submodule update --init         # 上流を取る（約 300 MB。モデルの重みと golden を含む）
./idf.sh build                      # 既定: モデル m3_decD_deep_full、PIE 有効
./idf.sh -p /dev/ttyACM0 flash
```

オプション:

| オプション | 意味 |
|---|---|
| `-DPF_PIE=0` | PIE を使わない参照実装。bit 一致の確認や陰性対照に |
| `-DPF_MODEL=m3_long_cfg` | 高速モデル（DiT 深さ 8、blob 2.57 MB）。切り替えるときは `build/` を消す |
| `-DPF_ARENA_INTERNAL=1` | 実験用。arena を内部 RAM に置く（`sdkconfig.arena_int` と組で使う）。いまは効果が無い |

## 使い方

起動すると自己テスト（約 0.1 秒）のあと、seed 3・8 ステップ・クラス 3（male / smile）・CFG w=6 で 1 枚生成します。

| 操作 | 動作 |
|---|---|
| 画像をタップ | seed を +1 して生成 |
| 右パネルの上半分をタップ | クラスを切り替え（female/no-smile → female/smile → male/no-smile → male/smile → null） |
| 右パネルの下半分をタップ | CFG を切り替え（none → 4 → 6 → 8 → none） |

### USB シリアル

115200 baud で、上流の Pico 版と同じ書式を受けます。

```
G <seed> [k_steps] [class] [w]     生成。class / w を省略すると golden の規約（class = seed % 5、w_idx = seed % 4 − 1）
I                                  モデル情報
```

応答は 1 行です。そのあとに段ごとの所要時間がログとして続きます。

```
OK seed=1 k=4 class=1 w=4 crc32=40c5e5a0 ms=2023
I (...) pico_faces:   DiT 1513 ms, VAE decode 498 ms (PIE)
  phase              calls      ms
  fn@4000b194       96    408.5
  ...
  stage(memcpy)     386    147.7  (18440 KB)
```

`fn@...` の関数名は `nm build/pico_faces_tab5.elf | grep -e _range -e conv_rows` で引けます。

⚠️ `cat /dev/ttyACM0` や `printf > /dev/ttyACM0` のようにポートを開閉すると、Linux の cdc-acm ドライバが
DTR/RTS を動かし、ESP32-P4 がダウンロードモードに落ちることがあります（実際に落ちました）。
`tools/serial_cmd.py` は開く前に DTR/RTS を下げるので安全です。

```bash
uv run --no-project --with pyserial python tools/serial_cmd.py --boot 14 --wait 10 "G 1 4" "G 2 4" "G 3 4"
```

## bit 一致の確認

次の 5 通りを打って同じ CRC32 が返れば、移植も PIE 版も bit 単位で正しいと言えます（実機で確認済み）。

| コマンド | crc32 |
|---|---|
| `G 1 4` | `40c5e5a0` |
| `G 2 4` | `630a7574` |
| `G 3 4` | `c76b6c2e` |
| `G 3 8 3 6` | `ecb7bc36` |
| `G 3 8 4 6` | `fd41f027` |

先頭 3 つは上流の golden そのものです。ホストでも再現できます。

```bash
gcc -O2 -Iupstream/checkpoints/m3_decD_deep_full -Iupstream/engine/include \
    upstream/engine/desktop/main_golden.c upstream/engine/src/*.c -o rf_golden
./rf_golden upstream/checkpoints/m3_decD_deep_full/model.bin out 4 1 2 3
cmp out/eng_1.rgb upstream/checkpoints/m3_decD_deep_full/goldens/golden_1.rgb
```

## 構成

```
CMakeLists.txt              PF_MODEL の選択、モデル blob の埋め込み
main/main.cpp               画面（M5GFX）・タッチ・USB シリアル・生成タスク
main/pf_par.c               rf_par_for を FreeRTOS の 2 タスク（コア 0 / 1）で実装。段ごとのプロファイル
components/pf_engine/       upstream/engine/src/*.c をそのままコンパイルするコンポーネント
components/pf_engine/pie/   ESP32-P4 PIE 版カーネル（次節）
tools/serial_cmd.py         USB シリアルにコマンドを送る補助
upstream/                   git submodule: cpldcpu/pico-faces（エンジン、モデル blob、golden）
sdkconfig.defaults          Tab5 の板設定（P4 v1.0、HEX PSRAM、USB Serial/JTAG、L2 キャッシュ 256 KB）
partitions.csv              app 7 MB（blob 4 MB を .rodata に含む）
idf.sh                      ESP-IDF v5.5 をクリーンな環境で有効にして idf.py を呼ぶ
```

### 板の層（`main/`）

- **モデル blob** は app の `.rodata` に埋め込み、起動時に PSRAM へコピーします。
- **2 コア並列** は上流の `firmware/par.c` と同じ契約（呼び出し側が前半、ワーカーが後半）を
  FreeRTOS の task notification で実装しました。生成タスクはコア 0、ワーカーはコア 1 に固定します。
  VAE デコーダのスクラッチが `rf_core_id()` で分かれているためです。
- **内部 RAM** には arena 256 KB と DiT の活性 88 KB が収まらないので、
  `components/pf_engine/linker.lf` でそれらの `.bss` を PSRAM に置いています。
  IDF 付属の `extram_bss` スキームは esp_wifi の fragment にあり、Wi-Fi を含まないビルドでは黙って無視されます。
  自前のスキーム（`bss -> extern_ram`）が必要でした。

### PIE の層（`components/pf_engine/pie/`）

上流のソースを変えずに差し替えるため、2 つの仕組みを使っています。

1. **影のヘッダ**。`make_shadow.py` が configure 時に上流の `rf_ops.h` から `build/pf_pie/rf_ops.h` を生成し、
   インクルードパスの先頭に置きます。差分は 3 箇所です（`RF_ALIGN4` を 16 整列に、参照実装の手前に
   `#elif defined(RF_PIE_P4)` で `rf_ops_pie.h` を挿入、`rf_rq` の 64 bit 乗算を `mulh` 1 命令に）。
   上流のヘッダが変わって当たらなくなると configure が止まります。
2. **リンカの `--wrap`**。別の翻訳単位から呼ばれるカーネルを `__wrap_` 版に差し替え、
   参照実装は `__real_` としてフォールバックと自己テストに残します。

| ファイル | 内容 |
|---|---|
| `rf_ops_pie.h` | `rf_dot_i8` 系の内積。XACC 累算（`esp.vmulas.s8.xacc`）、融合命令のソフトウェアパイプライン、非整列の x（`esp.ld.128.usar` + `esp.src.q`） |
| `pf_pie_linear.c` | 密な行列積（qkv / proj / fc1 / emb）。重みをブロック転置 `[O/16][K][16]` にして内部 SRAM の 200 KB にステージし、`esp.vsmulas.s8.qacc.ld.incp` で 16 出力を QACC に同時累算 |
| `pf_pie_kernels.c` | 疎な fc2（QACC gather）、rmsnorm（整数的に厳密な高速版）、起動時の自己テストとマイクロベンチ |
| `pf_pie_decode.c` | VAE デコーダ。疎な 3×3 畳み込みを QACC gather で（O=8 の層は 64 bit ロード）。重みは DiT と同じ内部 SRAM を時分割で使う |

**自己テスト**は起動時に走ります。内積（K 6 種 × 非整列 16 通り）、`rf_rq`（4,000 組）、rmsnorm（300 組）、
isqrt / 除算（3,000 組）、列方向行列積（4 形状 × 3 エピローグ）、fc2 gather、VAE gather を参照実装と突き合わせ、
1 つでも食い違えば PIE を切ってすべてスカラに戻します（遅いが正しい）。

## 測定で分かったこと

### ESP32-P4 の PIE（400 MHz）

| 項目 | 実測 |
|---|---|
| PIE 命令 1 個（MAC でもロードでも） | 約 3.3 ns ≈ 1.3 サイクル。ロードと MAC は重ならない |
| MAC + ロードの融合命令 | 約 6.1 ns（2 命令ぶん。命令数は減るが時間は減らない） |
| 16 MAC あたりの下限 | 約 6.2 ns（重みロード 1 + MAC 1） |
| `esp.movx.r.xacc.l` | 下位 **24 bit** しか返さない。`.h` と結合して 32 bit を組み立てる |
| QACC のレーン順（s8 MAC） | `st.qacc.l.l / l.h / h.l / h.h` の順に書き出すと int32[16] の恒等 |
| PSRAM 上の重み | L1 ミス時に L2 のレイテンシで PIE のロードが止まり、1 行 19 ns。内部 SRAM に置くと 2〜3 倍速い |

PIE 命令のオペランドは x8–x15 / x24–x31 しか取れないので `register ... asm("a2")` で渡しています。
初期化から asm までの間に関数呼び出しを挟むと壊れます（起動時に Load access fault で再起動を繰り返す事故を 1 回起こしました）。

### 段ごとの内訳（K=4 w=4、2.02 s）

| 段 | ms | 次の一手 |
|---|---:|---|
| mlp（fc1 + 疎 fc2） | 600 | fc2 の gather を L1 に収まる単位に分け、QACC を退避 / 復帰する |
| norm_qkv | 409 | rmsnorm の固定コスト（isqrt・除算）をさらに削る |
| VAE 疎な畳み込み | 390 | 同上の分割 |
| attention | 256 | softmax の 64 回の除算を逆数乗算（厳密）に |
| 重みステージング memcpy | 148 | `esp_async_memcpy`（GDMA）で計算と重ねる |
| VAE 密な畳み込み（C=8） | 104 | スカラのまま。im2col + PIE 内積 |
| proj | 72 | |

## 既知の問題

- 起動後しばらく、触っていないのにタッチの「クリック」が来て勝手に生成が始まることがあります。
  起動 3 秒間は無視していますが、その後にも数秒おきに来た回がありました。座標をログに出してあります。

## ライセンス

上流の cpldcpu/pico-faces にはライセンス表記がありません（2026-09-06 時点）。この移植の `main/` と
`components/` は上流に依存しており、再配布や公開の可否は上流の作者の意向によります。
