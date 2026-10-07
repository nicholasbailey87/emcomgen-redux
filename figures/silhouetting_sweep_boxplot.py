"""
Box plot of final test accuracy against `silhouette_p_receiver` for
`experiments/silhouetting_sweep`.

Each rate gets three boxes -- shape (`test_acc_md_shape`), colour
(`test_acc_md_color`) and overall (`test_acc`) -- read at the last epoch of each
seed's CSV, with the seeds overlaid as dots since three seeds make a thin box.

Reads `results/silhouetting_sweep/<rate>_seed_<n>.csv` beside this file and
writes `silhouetting_sweep_boxplot.png` beside it.

    venv/bin/python figures/silhouetting_sweep_boxplot.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "silhouetting_sweep"
OUTPUT = HERE / "silhouetting_sweep_boxplot.png"

SERIES = [
    ("test_acc_md_shape", "Shape", "#7B3FA0"),
    ("test_acc_md_color", "Colour", "#1F6FD1"),
    ("test_acc", "Overall", "#111111"),
]
BOX_WIDTH = 0.22


def final_rows() -> pd.DataFrame:
    rows = []
    for path in sorted(RESULTS.glob("*_seed_*.csv")):
        rate = float(path.name.split("_")[0])
        last = pd.read_csv(path).iloc[-1]
        rows.append({"rate": rate, **{k: last[k] for k, _, _ in SERIES}})
    return pd.DataFrame(rows)


def main():
    finals = final_rows()
    rates = sorted(finals.rate.unique())

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, (key, label, colour) in enumerate(SERIES):
        positions = np.arange(len(rates)) + (i - 1) * BOX_WIDTH
        data = [finals[finals.rate == r][key].values for r in rates]
        ax.boxplot(
            data,
            positions=positions,
            widths=BOX_WIDTH * 0.8,
            patch_artist=True,
            boxprops=dict(facecolor=colour + "33", edgecolor=colour, linewidth=1.5),
            medianprops=dict(color=colour, linewidth=2),
            whiskerprops=dict(color=colour),
            capprops=dict(color=colour),
            showfliers=False,
        )
        for x, values in zip(positions, data):
            ax.scatter(np.full(len(values), x), values, color=colour, s=18, zorder=3)
        ax.plot([], [], color=colour, lw=6, alpha=0.5, label=label)

    ax.set_xticks(range(len(rates)))
    ax.set_xticklabels([f"{r:.1f}" for r in rates])
    ax.set_xlabel("Silhouetting amount")
    ax.set_ylabel("Final test accuracy")
    ax.axhline(0.5, color="grey", ls=":", lw=1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=150)


if __name__ == "__main__":
    main()
