"""Plot each policy's success rate during training from the TensorBoard logs.

Policy 1 (pluck): share of episodes that plucked the apple with a grip (Pluck/plucked_with_grip).
Policy 2 (place): share of episodes that met harvest_isaac's full success rule (Place/success).
"""

import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

LOGS = r"C:\isaac\IsaacLab3\logs\rsl_rl"
RUNS = [  # (label, run folder glob, metric, colour) - categorical slots 1 and 2
    ("Policy 1: plucked with grip", r"harvest_pluck\*pluck_scratch", "Pluck/plucked_with_grip", "#2a78d6"),
    ("Policy 2: full success (pluck, place, release)", r"harvest_place\*place_from_pluck400", "Place/success",
     "#eb6834"),
]
SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"


def series(pattern, tag):
    run = sorted(glob.glob(os.path.join(LOGS, pattern)))[-1]
    ea = EventAccumulator(run, size_guidance={"scalars": 0})
    ea.Reload()
    ev = ea.Scalars(tag)
    return [e.step for e in ev], [100.0 * e.value for e in ev]


def ema(values, alpha=0.05):
    out, s = [], values[0]
    for v in values:
        s = alpha * v + (1 - alpha) * s
        out.append(s)
    return out


fig, ax = plt.subplots(figsize=(10, 5.2), dpi=150)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
for label, pattern, tag, colour in RUNS:
    steps, vals = series(pattern, tag)
    ax.plot(steps, vals, color=colour, linewidth=1, alpha=0.22)
    smooth = ema(vals)
    ax.plot(steps, smooth, color=colour, linewidth=2, solid_capstyle="round", label=label)
    ax.annotate(f"{smooth[-1]:.0f}%", (steps[-1], smooth[-1]), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=10, color=TEXT)

ax.set_ylim(-3, 105)
ax.axvline(400, color=TEXT2, linewidth=1)
ax.annotate("Policy 2 starts from\npluck model_400", (400, 104), xytext=(6, -2),
            textcoords="offset points", va="top", fontsize=9, color=TEXT2)

ax.set_title("Success rate during training", loc="left", fontsize=14, color=TEXT, pad=26)
ax.text(0, 1.02, "Share of finished episodes per iteration, with exploration noise. "
        "Thick line: smoothed (EMA 0.05); faint line: per iteration.",
        transform=ax.transAxes, fontsize=9, color=TEXT2)
ax.set_xlabel("Training iteration", color=TEXT2)
ax.set_ylabel("Success rate", color=TEXT2)
ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
ax.grid(axis="y", color=GRID, linewidth=1)
ax.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(colors=TEXT2, length=0)
leg = ax.legend(loc="center", bbox_to_anchor=(0.5, 0.5), frameon=False, fontsize=10)
for t in leg.get_texts():
    t.set_color(TEXT)
fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "success_rate.png")
fig.savefig(out, facecolor=SURFACE)
print(out)
