"""
Compose 2x2 appendix grid from pre-rendered PNGs under appendix_promptad_minimal/.

Does not recompute experiments. Run from repo root:

    python PromptAD/utils/make_promptad_appendix_grid.py
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


def _minimal_dir(custom: str | None) -> str:
    if custom:
        return os.path.abspath(custom)
    return os.path.abspath(os.path.join(_repo_root(), "PromptAD", "appendix_promptad_minimal"))


def pick_panel_a_path(minimal: str) -> str | None:
    candidates = [
        os.path.join(minimal, "AUROC_vs_instability_all.png"),
        os.path.join(minimal, "fig_s1_spearman_histogram.png"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--minimal_dir",
        type=str,
        default=None,
        help="Override path to appendix_promptad_minimal",
    )
    args = parser.parse_args()

    minimal = _minimal_dir(args.minimal_dir)
    out_png = os.path.join(minimal, "promptad_appendix_grid.png")
    out_pdf = os.path.join(minimal, "promptad_appendix_grid.pdf")

    paths = [
        pick_panel_a_path(minimal),
        os.path.join(minimal, "extra_category", "instability_distribution.png"),
        os.path.join(minimal, "extra_category", "instability_vs_error.png"),
        os.path.join(minimal, "extra_category", "rejection_curve.png"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    letters = ("a", "b", "c", "d")

    for ax, letter, p in zip(axes.flat, letters, paths):
        ax.axis("off")
        if p is not None and os.path.isfile(p):
            img = mpimg.imread(p)
            ax.imshow(img, aspect="auto", interpolation="nearest")
        else:
            hint = os.path.basename(p) if p else "panel image"
            ax.text(
                0.5,
                0.5,
                f"missing:\n{hint}",
                ha="center",
                va="center",
                fontsize=9,
                color="0.4",
            )
        ax.text(
            0.03,
            0.97,
            f"({letter})",
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            va="top",
            ha="left",
            color="0.05",
            zorder=10,
        )

    plt.tight_layout(pad=0.35, h_pad=0.15, w_pad=0.15)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print("Wrote:", out_png)
    print("Wrote:", out_pdf)
    print("(a) source:", paths[0] or "(none found)")


if __name__ == "__main__":
    main()
