"""
Compose figure1_evidence/*.png into one row with panel labels (a)(b)(c) below each image.

    python PromptAD/utils/make_figure1_evidence_row.py
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evidence_dir",
        type=str,
        default=None,
        help="Default: PromptAD/appendix_promptad_minimal/figure1_evidence",
    )
    args = parser.parse_args()

    ev = args.evidence_dir or os.path.join(
        _repo_root(), "PromptAD", "appendix_promptad_minimal", "figure1_evidence"
    )
    ev = os.path.abspath(ev)

    paths = [
        os.path.join(ev, "fig_figure1_c_spearman_distribution.png"),
        os.path.join(ev, "fig_figure1_d_rel_drop_histogram.png"),
        os.path.join(ev, "fig_figure1_d_improved_counts.png"),
    ]
    for p in paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(p)

    letters = ("a", "b", "c")
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))

    for ax, path, letter in zip(axes, paths, letters):
        ax.imshow(mpimg.imread(path), aspect="auto", interpolation="nearest")
        ax.axis("off")
        ax.text(
            0.5,
            -0.06,
            f"({letter})",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=12,
            fontweight="bold",
            color="0.05",
            clip_on=False,
        )

    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.14, wspace=0.08)
    out_png = os.path.join(ev, "figure1_evidence_row_abc.png")
    out_pdf = os.path.join(ev, "figure1_evidence_row_abc.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    print("Wrote:", out_png)
    print("Wrote:", out_pdf)


if __name__ == "__main__":
    main()
