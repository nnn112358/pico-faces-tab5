"""推論速度のグラフ（横向きの grouped bar）。dataviz スキルの手順: 形 → 色（2 色、検証済み）→ 細いマーク・
2 px の隙間・直接ラベル → 凡例（4 系列）。色は板ごと（CoreS3 = orange、Tab5 = blue。scripts/validate_palette.js で
light モードの全チェック PASS）。PIE なし / あり は同じ色でハッチの有無（テクスチャ）で分け、値は全部直接ラベル。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import sys

plt.rcParams["font.family"] = "IPAexGothic"
SURF, TXT, TXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e6e3"
ORANGE, BLUE = "#eb6834", "#2a78d6"   # slot 2 / slot 1（検証済み）

settings = ["K=4, cfg none", "K=4, cfg w=4\n(golden の規約)", "K=8, cfg none\n(起動時の既定)", "K=8, cfg w=6"]
series = [  # (label, color, hatch, values)
    ("CoreS3  PIE なし（参照 C）", ORANGE, "///", [17.0, 27.9, 27.9, 49.7]),
    ("CoreS3  PIE あり",           ORANGE, None,  [4.9, 6.9, 6.9, 10.9]),
    ("Tab5  PIE なし（参照 C）",   BLUE,   "///", [5.99, 10.11, 10.09, 18.30]),
    ("Tab5  PIE あり",             BLUE,   None,  [1.27, 2.02, 2.03, 3.52]),
]

fig, ax = plt.subplots(figsize=(12, 6.6), dpi=160)
fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
n = len(series); h = 0.17; gap = 0.03
ys = []
for gi in range(len(settings)):
    base = gi * 1.0
    for si, (label, color, hatch, vals) in enumerate(series):
        y = base + (si - (n - 1) / 2) * (h + gap)
        v = vals[gi]
        if hatch:
            ax.barh(y, v, height=h, color=SURF, edgecolor=color, linewidth=1.2, hatch=hatch, zorder=3)
        else:
            ax.barh(y, v, height=h, color=color, edgecolor=color, linewidth=0.8, zorder=3)
        ax.text(v + 0.6, y, f"{v:.1f} s" if v >= 10 else f"{v:.2f} s", va="center", ha="left", fontsize=10, color=TXT, zorder=4)
    ys.append(base)

# 上流 RP2350（K=4 w=4 のみ公称 約 10 s）を目安として置く
ax.plot([10, 10], [ys[1] - 0.42, ys[1] + 0.42], color=TXT2, linewidth=1.2, linestyle=(0, (3, 3)), zorder=2)
ax.text(10.4, ys[1] + 0.47, "上流の RP2350 @300 MHz ≈ 10 s", fontsize=9, color=TXT2, va="bottom", ha="left")

ax.set_yticks(ys); ax.set_yticklabels(settings, fontsize=11, color=TXT)
ax.invert_yaxis()
ax.set_xlim(0, 56); ax.set_xlabel("1 枚の生成時間 [秒]（短いほど速い）", fontsize=11, color=TXT2)
ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0); ax.set_axisbelow(True)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(axis="x", colors=TXT2, labelsize=10); ax.tick_params(axis="y", length=0)

handles = [Patch(facecolor=(SURF if hatch else c), edgecolor=c, hatch=hatch or "", linewidth=1.2, label=l) for l, c, hatch, _ in series]
ax.legend(handles=handles, loc="lower right", fontsize=10, frameon=False, labelcolor=TXT, ncol=2)
fig.suptitle("pico-faces の 1 枚あたりの生成時間（既定モデル m3_decD_deep_full）", fontsize=14, color=TXT, x=0.012, ha="left", y=0.985)
ax.set_title("M5Stack CoreS3（ESP32-S3 240 MHz × 2）と M5Stack Tab5（ESP32-P4 360 MHz × 2）。PIE なし / あり で画像は bit 一致（CRC32 が同じ）",
             fontsize=9.5, color=TXT2, loc="left", pad=10)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = sys.argv[1]
fig.savefig(out, facecolor=SURF)
print("wrote", out)
