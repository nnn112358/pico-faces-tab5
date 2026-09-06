"""Tab5 の USB シリアルにコマンドを送り、応答を集める（DTR/RTS を動かさない）。

    uv run --no-project --with pyserial python tools/serial_cmd.py [--port /dev/ttyACM0]
        [--boot 25] [--wait 12] "G 1 4" "G 2 4" ...

⚠️ `cat /dev/ttyACM0` や `printf > /dev/ttyACM0` のようにポートを開閉すると、Linux の
cdc-acm ドライバが DTR/RTS を動かし、ESP32-P4 の USB-Serial-JTAG がダウンロードモードに
落ちることがある（実際に落ちた）。pyserial で開く前に dtr/rts を下げておけば起きない。

--boot N : 最初に N 秒だけ起動ログを集める（0 で省略）
--wait N : 各コマンドの応答を最大 N 秒待つ（`OK seed=` の行が出たら早く切り上げる）
"""
import argparse
import sys
import time

import serial

ap = argparse.ArgumentParser()
ap.add_argument("--port", default="/dev/ttyACM0")
ap.add_argument("--boot", type=float, default=0)
ap.add_argument("--wait", type=float, default=12)
ap.add_argument("--reset", action="store_true", help="DTR/RTS でリセットしてから始める")
ap.add_argument("cmds", nargs="*")
a = ap.parse_args()

ser = serial.Serial()
ser.port = a.port
ser.baudrate = 115200
ser.timeout = 0.2
ser.dtr = False
ser.rts = False
for attempt in range(20):   # 書き込み直後は再列挙で数秒消えることがある
    try:
        ser.open()
        break
    except serial.SerialException as e:
        if attempt == 19:
            sys.exit(f"serial_cmd.py: cannot open {a.port}: {e}")
        time.sleep(0.5)

def collect(seconds, stop_prefix=None):
    t_end = time.time() + seconds
    buf = b""
    while time.time() < t_end:
        try:
            chunk = ser.read(4096)
        except serial.SerialException as e:   # 読み取り中の一時的なエラーは待って続ける
            sys.stderr.write(f"serial_cmd.py: read error, retrying: {e}\n")
            time.sleep(0.3)
            continue
        if chunk:
            buf += chunk
            if stop_prefix and stop_prefix in buf:
                # 応答の後に続くプロファイル表を少しだけ待つ
                time.sleep(0.3)
                buf += ser.read(65536)
                break
    return buf.decode("utf-8", "replace")

if a.reset:
    ser.dtr = False; ser.rts = True; time.sleep(0.1); ser.rts = False; time.sleep(0.1)
if a.boot > 0:
    print("=== boot"); print(collect(a.boot, b"OK seed=" if not a.cmds else None), end="")
for c in a.cmds:
    ser.reset_input_buffer()
    ser.write((c + "\n").encode()); ser.flush()
    print(f"=== {c}"); print(collect(a.wait, b"OK seed="), end="")
ser.close()
