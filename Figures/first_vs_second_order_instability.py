import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# Model A: stable — half pairs always correct, half always wrong
A = np.array(
    [
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ],
    dtype=float,
)

# Model B: unstable — same mean correctness as A
B = np.array(
    [
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 0, 1],
        [0, 1, 1, 0],
    ],
    dtype=float,
)

out_dir = "Figures"
os.makedirs(out_dir, exist_ok=True)

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    }
)

# 0 = black (incorrect), 1 = white (correct)
cmap = ListedColormap(["#000000", "#ffffff"])

fig, axes = plt.subplots(1, 2, figsize=(6, 2.6))

n_rows, n_cols = A.shape
pair_labels = [f"{i + 1}" for i in range(n_rows)]
cond_labels = [f"{j + 1}" for j in range(n_cols)]

for ax, data, title in zip(
    axes,
    (A, B),
    ("Model A: Stable", "Model B: Unstable"),
):
    ax.imshow(data, cmap=cmap, vmin=0, vmax=1, interpolation="nearest", aspect="equal")
    ax.set_title(title, fontweight="bold", pad=8)
    ax.set_xlabel("Equivalent conditions (views)")
    ax.set_ylabel("Pairs")
    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(cond_labels)
    ax.set_yticklabels(pair_labels)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_edgecolor("black")
    ax.grid(False)

plt.tight_layout(rect=[0, 0.22, 1, 1])
fig.text(
    0.5,
    0.14,
    "Same AUROC (first-order)",
    ha="center",
    va="top",
    fontsize=13,
    fontweight="bold",
)
fig.text(
    0.5,
    0.04,
    "Different instability (second-order)",
    ha="center",
    va="top",
    fontsize=13,
    fontweight="bold",
)

pdf_path = os.path.join(out_dir, "first_vs_second_order_instability.pdf")
png_path = os.path.join(out_dir, "first_vs_second_order_instability.png")
fig.savefig(pdf_path, bbox_inches="tight", facecolor="white", edgecolor="none")
fig.savefig(png_path, bbox_inches="tight", facecolor="white", edgecolor="none")
plt.close(fig)
