# pico-faces on M5Stack Tab5

[cpldcpu/pico-faces](https://github.com/cpldcpu/pico-faces) を **M5Stack Tab5（ESP32-P4）** で動かすファームウェアです。

## 元のリポジトリについて

pico-faces は、Raspberry Pi Pico 2（RP2350）のようなマイコンで顔画像を生成する小さな拡散モデルです。

- 潜在空間で動く拡散トランスフォーマー（DiT、約 2.4M パラメータ）と、潜在を 128×128 の RGB 画像に戻す VAE デコーダから成ります。
- 重みも計算もすべて int8 の整数演算で、推論エンジンは移植しやすい C99 で書かれています。
- 条件は「性別 × 笑顔」の 4 クラスと無条件の計 5 つで、classifier-free guidance（CFG）の強さも選べます。
- 生成は seed ごとに決定的で、同じ seed なら PC でもマイコンでも 1 bit も違わない画像になります。

この移植では、その推論エンジンを**無改変**のまま ESP-IDF でビルドし、Tab5 の画面・タッチ・USB シリアルと、
ESP32-P4 の SIMD 命令（PIE）による高速化を足しています。

## 使い方

### 用意するもの

- M5Stack Tab5
- ESP-IDF v5.5（`~/esp/esp-idf` に展開してあるものを `idf.sh` が使います）
- `uv`（ビルド中に Python スクリプトを 1 本走らせます）

### ビルドと書き込み

```bash
git submodule update --init         # 上流を取る（約 300 MB。モデルの重みを含む）
./idf.sh build
./idf.sh -p /dev/ttyACM0 flash
```

### 操作

起動すると seed 3 で 1 枚生成して表示します（8 ステップ、male / smile、CFG w=6）。
右パネルのボタンで seed と条件を選んで生成します。

| ボタン | 動作 |
|---|---|
| `-10` `-1` `+1` `+10` | 次に生成する seed を増減（中央に表示） |
| `class : ...` | クラスを切り替え（female/no-smile → female/smile → male/no-smile → male/smile → null） |
| `cfg : ...` | CFG を切り替え（none → w=4 → 6 → 8 → none） |
| **1枚生成** | 表示中の seed で 1 枚生成し、seed を 1 進める（左の画像をタップしても同じ） |
| **10枚連続** | seed から 10 枚を順に生成し、seed を 10 進める。途中で画面を触ると中断 |

seed と条件が同じなら毎回同じ顔になります（生成は決定的です）。別の顔にしたいときは seed を変えてください。

USB シリアル（115200）からは上流の Pico 版と同じ書式で指示できます。

```
G <seed> [k_steps] [class] [w] [count]   生成。class / w を省略すると上流の golden の規約。
                                         count（既定 1）枚を seed から順に生成
I                                        モデル情報
```

応答は `OK seed=1 k=4 class=1 w=4 crc32=40c5e5a0 ms=2023` のような 1 行です。
`crc32` が上流の golden と一致すれば bit 単位で正しく動いています（値は [docs/details.md](docs/details.md)）。

⚠️ `cat` や `printf` でシリアルポートを開閉すると ESP32-P4 がダウンロードモードに落ちることがあります。
`tools/serial_cmd.py` を使ってください。

```bash
uv run --no-project --with pyserial python tools/serial_cmd.py --boot 14 --wait 10 "G 1 4"
```

## 推論速度

Tab5（ESP32-P4 360 MHz × 2 コア）での 1 枚の生成時間です。PIE 版も参照実装と bit 一致します。

| 設定 | 参照 C（移植直後） | PIE 版（現在） |
|---|---:|---:|
| K=4、CFG w=4 | 10.10 s | **2.02 s** |
| K=8、CFG なし | 10.08 s | **2.03 s** |
| K=8、CFG w=6 | 18.36 s | **3.53 s** |

上流の RP2350 @300 MHz は K=4 w=4 で約 10 秒です。
`-DPF_PIE=0` を付けてビルドすると参照実装に戻ります。

## 詳しい内容

構成、PIE 版カーネルの仕組み、自己テスト、bit 一致の確認手順、測定結果、既知の問題は
[docs/details.md](docs/details.md) にまとめてあります。

## ライセンス

上流の cpldcpu/pico-faces にはライセンス表記がありません（2026-09-06 時点）。この移植は上流に依存しており、
再配布や公開の可否は上流の作者の意向によります。
