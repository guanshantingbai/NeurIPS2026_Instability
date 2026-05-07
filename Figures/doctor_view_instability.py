import os

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# 0.00 -> 1.00: white -> light gray -> dark gray -> black
cmap_score = LinearSegmentedColormap.from_list(
    "white_to_black_gray",
    ["#ffffff", "#c8c8c8", "#505050", "#000000"],
    N=256,
)

model_a = np.array(
    [
        [0.90, 0.10, 0.60, 0.40],
        [0.91, 0.11, 0.61, 0.39],
    ],
    dtype=float,
)

model_b = np.array(
    [
        [0.55, 0.45, 0.60, 0.40],
        [0.55, 0.35, 0.50, 0.40],
    ],
    dtype=float,
)

col_labels = ["A1", "A2", "N1", "N2"]
row_labels = ["Machine A", "Machine B"]

out_dir = "Figures"
os.makedirs(out_dir, exist_ok=True)

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    }
)

fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))

for ax_idx, (ax, data) in enumerate(zip(axes, (model_a, model_b))):
    ax.imshow(
        data,
        cmap=cmap_score,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="equal",
    )
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_edgecolor("black")
    ax.grid(False)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            txt_color = "white" if val >= 0.52 else "black"
            outline = "black" if txt_color == "white" else "white"
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color=txt_color,
                fontsize=13,
                fontweight="bold",
                path_effects=[pe.withStroke(linewidth=2.0, foreground=outline)],
            )

    ax.text(
        0.5,
        -0.14,
        f"({chr(ord('a') + ax_idx)})",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=13,
        fontweight="bold",
    )

plt.tight_layout(rect=[0, 0.08, 1, 1])

pdf_path = os.path.join(out_dir, "doctor_view_instability.pdf")
png_path = os.path.join(out_dir, "doctor_view_instability.png")
fig.savefig(pdf_path, bbox_inches="tight", facecolor="white", edgecolor="none")
fig.savefig(png_path, bbox_inches="tight", facecolor="white", edgecolor="none")
plt.close(fig)
