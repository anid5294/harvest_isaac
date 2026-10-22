"""Plot mean episode reward (TensorBoard Train/mean_reward) for the pluck and place runs."""

import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

LOGS = r"C:\isaac\IsaacLab3\logs\rsl_rl"
RUNS = [  # (label, run folder glob, colour) - categorical slots 1 and 2
    ("Policy 1: pluck (from scratch)", r"harvest_pluck\*pluck_scratch", "#2a78d6"),
    ("Policy 2: place (from pluck model_400)", r"harvest_place\*place_from_pluck400", "#eb6834"),
]
SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"


def series(pattern):
    run = sorted(glob.glob(os.path.join(LOGS, pattern)))[-1]
    ea = EventAccumulator(run, size_guidance={"scalars": 0})
    ea.Reload()
    ev = ea.Scalars("Train/mean_reward")
    return [e.step for e in ev], [e.value for e in ev]


def ema(values, alpha=0.05):
    out, s = [], values[0]
    for v in values:
        s = alpha * v + (1 - alpha) * s
        out.append(s)
    return out


fig, ax = plt.subplots(figsize=(10, 5.2), dpi=150)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
for label, pattern, colour in RUNS:
    steps, vals = series(pattern)
    ax.plot(steps, vals, color=colour, linewidth=1, alpha=0.22)
    smooth = ema(vals)
    ax.plot(steps, smooth, color=colour, linewidth=2, solid_capstyle="round", label=label)
    ax.annotate(f"{smooth[-1]:.0f}", (steps[-1], smooth[-1]), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=10, color=TEXT)

ax.axvline(400, color=TEXT2, linewidth=1)
ax.annotate("Policy 2 starts from\npluck model_400", (400, ax.get_ylim()[1]), xytext=(6, -6),
            textcoords="offset points", va="top", fontsize=9, color=TEXT2)

ax.set_title("Mean episode reward during training", loc="left", fontsize=14, color=TEXT, pad=26)
ax.text(0, 1.02, "Thick line: smoothed (EMA 0.05); faint line: per iteration. "
        "The two policies use different reward functions, so their scales differ.",
        transform=ax.transAxes, fontsize=9, color=TEXT2)
ax.set_xlabel("Training iteration", color=TEXT2)
ax.set_ylabel("Mean episode reward", color=TEXT2)
ax.grid(axis="y", color=GRID, linewidth=1)
ax.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(colors=TEXT2, length=0)
leg = ax.legend(loc="lower right", frameon=False, fontsize=10)
for t in leg.get_texts():
    t.set_color(TEXT)
fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "total_reward.png")
fig.savefig(out, facecolor=SURFACE)
print(out)
