"""
Draw PaDiM panels (b)(c)(d) with PromptAD-identical matplotlib styling.

Panel (a) for the optional 2×2 composite uses `draw_panel_a` from padim_build_panel_a
(same Instability Score label as standalone panel a).

Requires (full b+c+d): panel (a) CSV with `flip_rate_mean` + exp5 CSVs from
`padim_exp5_visa_macaroni1.py` under .../experiments/.

`--only_c_d`: only exp5 CSVs (VisA macaroni1, ResNet-18); no panel CSV / WR50 wait.

`--only_b`: only panel (b) from panel CSV `flip_rate_mean` (no exp5 required).
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from padim_build_panel_a import draw_panel_a
from padim_figure_promptad_style import (
    PNG_DPI,
    PANEL_LETTER_FS,
    TITLE_FS,
    draw_panel_b_flip_hist,
    draw_panel_c_instability_error,
    draw_panel_d_rejection,
    draw_panel_d_seed_risk_coverage,
)
from padim_seed_killer_evidence_pipeline import PROXY_COL as SEED_KILLER_PROXY_COL


def main() -> None:
    repo = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(repo, "result_analysis", "figures")
    default_csv = os.path.join(fig_dir, "padim_panel_a_mvtec_visa_r18_wr50.csv")
    default_exp5 = os.path.join(repo, "result_analysis", "exp5_visa_macaroni1_r18", "experiments")

    p = argparse.ArgumentParser()
    p.add_argument("--panel_a_csv", type=str, default=default_csv)
    p.add_argument("--exp5_experiments_dir", type=str, default=default_exp5)
    p.add_argument("--out_dir", type=str, default=fig_dir)
    p.add_argument(
        "--killer_pair_json",
        type=str,
        default=os.path.join(repo, "padim_result_seed_search", "analysis", "killer_pair.json"),
        help="Panel (d) in 2x2: same seeds/slug as killer_final.png (risk–coverage redraw).",
    )
    p.add_argument(
        "--seed_search_root",
        type=str,
        default=os.path.join(repo, "padim_result_seed_search"),
        help="Directory containing <slug>/<seed>/per_sample.csv.",
    )
    p.add_argument(
        "--seed_proxy",
        type=str,
        default="u6",
        choices=sorted(SEED_KILLER_PROXY_COL.keys()),
        help="Ordering proxy for panel (d) curves (match phase3 --proxy).",
    )
    p.add_argument("--no_abcd_grid", action="store_true", help="Skip 2x2 composite PNG")
    p.add_argument(
        "--only_c_d",
        action="store_true",
        help="Only panels (c) and (d): VisA macaroni1 + ResNet-18 exp5 CSVs only; no panel (a) CSV / flip_rate_mean.",
    )
    p.add_argument(
        "--only_b",
        action="store_true",
        help="Only panel (b) from panel CSV flip_rate_mean (no exp5 files needed).",
    )
    args = p.parse_args()

    if args.only_c_d and args.only_b:
        raise SystemExit("Choose at most one of --only_b and --only_c_d.")

    os.makedirs(args.out_dir, exist_ok=True)

    if args.only_b:
        df = pd.read_csv(args.panel_a_csv)
        if "dataset" not in df.columns:
            df["dataset"] = "mvtec"
        if "flip_rate_mean" not in df.columns:
            raise SystemExit(f"Missing flip_rate_mean in {args.panel_a_csv}")
        if df["flip_rate_mean"].notna().sum() == 0:
            raise SystemExit("All flip_rate_mean are NaN.")
        fig_b, ax_b = plt.subplots(figsize=(4.8, 3.9))
        draw_panel_b_flip_hist(ax_b, df)
        ax_b.set_title("Distribution of instability", fontsize=TITLE_FS)
        p_b = os.path.join(args.out_dir, "padim_panel_b_flip_hist.png")
        fig_b.tight_layout()
        fig_b.savefig(p_b, dpi=300, bbox_inches="tight", pad_inches=0.04)
        plt.close(fig_b)
        print("Wrote", p_b)
        return

    sample_csv = os.path.join(args.exp5_experiments_dir, "exp5_sample_ranking_error.csv")
    rej_csv = os.path.join(args.exp5_experiments_dir, "exp5_instability_rejection.csv")
    if not os.path.isfile(sample_csv):
        raise FileNotFoundError(f"Run padim_exp5_visa_macaroni1.py first. Missing {sample_csv}")
    if not os.path.isfile(rej_csv):
        raise FileNotFoundError(f"Missing {rej_csv}")

    if args.only_c_d:
        fig_c, ax_c = plt.subplots(figsize=(4.8, 3.9))
        draw_panel_c_instability_error(ax_c, sample_csv)
        ax_c.set_title("Instability vs error", fontsize=TITLE_FS)
        p_c = os.path.join(args.out_dir, "padim_panel_c_macaroni1_r18.png")
        fig_c.tight_layout()
        fig_c.savefig(p_c, dpi=300, bbox_inches="tight", pad_inches=0.04)
        plt.close(fig_c)
        print("Wrote", p_c)

        fig_d, ax_d = plt.subplots(figsize=(4.8, 3.9))
        draw_panel_d_rejection(ax_d, rej_csv)
        ax_d.set_title("Instability-aware rejection", fontsize=TITLE_FS)
        p_d = os.path.join(args.out_dir, "padim_panel_d_macaroni1_r18.png")
        fig_d.tight_layout()
        fig_d.savefig(p_d, dpi=300, bbox_inches="tight", pad_inches=0.04)
        plt.close(fig_d)
        print("Wrote", p_d)
        return

    df = pd.read_csv(args.panel_a_csv)
    if "dataset" not in df.columns:
        df["dataset"] = "mvtec"

    if "flip_rate_mean" not in df.columns:
        raise SystemExit(
            f"Missing column flip_rate_mean in {args.panel_a_csv}. "
            "Re-run padim_build_panel_a.py (full or per-row) to populate it."
        )
    n_flip = df["flip_rate_mean"].notna().sum()
    if n_flip == 0:
        raise SystemExit(
            "All flip_rate_mean values are NaN. Recompute panel rows with the updated padim_build_panel_a.py."
        )

    # --- Standalone (b)(c)(d) ---
    fig_b, ax_b = plt.subplots(figsize=(4.8, 3.9))
    draw_panel_b_flip_hist(ax_b, df)
    ax_b.set_title("Distribution of instability", fontsize=TITLE_FS)
    p_b = os.path.join(args.out_dir, "padim_panel_b_flip_hist.png")
    fig_b.tight_layout()
    fig_b.savefig(p_b, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig_b)
    print("Wrote", p_b)

    fig_c, ax_c = plt.subplots(figsize=(4.8, 3.9))
    draw_panel_c_instability_error(ax_c, sample_csv)
    ax_c.set_title("Instability vs error", fontsize=TITLE_FS)
    p_c = os.path.join(args.out_dir, "padim_panel_c_macaroni1_r18.png")
    fig_c.tight_layout()
    fig_c.savefig(p_c, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig_c)
    print("Wrote", p_c)

    fig_d, ax_d = plt.subplots(figsize=(4.8, 3.9))
    draw_panel_d_rejection(ax_d, rej_csv)
    ax_d.set_title("Instability-aware rejection", fontsize=TITLE_FS)
    p_d = os.path.join(args.out_dir, "padim_panel_d_macaroni1_r18.png")
    fig_d.tight_layout()
    fig_d.savefig(p_d, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig_d)
    print("Wrote", p_d)

    if args.no_abcd_grid:
        return

    titles = (
        "AUROC vs instability",
        "Distribution of instability",
        "Instability vs error",
        "Same AUROC, different decision reliability",
    )
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    draw_panel_a(axes[0, 0], df)
    axes[0, 0].set_title(titles[0], fontsize=TITLE_FS)

    draw_panel_b_flip_hist(axes[0, 1], df)
    axes[0, 1].set_title(titles[1], fontsize=TITLE_FS)

    draw_panel_c_instability_error(axes[1, 0], sample_csv)
    axes[1, 0].set_title(titles[2], fontsize=TITLE_FS)

    draw_panel_d_seed_risk_coverage(
        axes[1, 1],
        killer_json_path=os.path.abspath(args.killer_pair_json),
        search_root=os.path.abspath(args.seed_search_root),
        proxy_key=args.seed_proxy,
    )
    axes[1, 1].set_title(titles[3], fontsize=TITLE_FS)

    for ax, letter in zip(axes.flat, ("a", "b", "c", "d")):
        ax.text(
            -0.12,
            1.05,
            f"({letter})",
            transform=ax.transAxes,
            fontsize=PANEL_LETTER_FS,
            fontweight="bold",
            ha="left",
            va="bottom",
            color="0.1",
            zorder=10,
            clip_on=False,
        )

    plt.subplots_adjust(left=0.13, right=0.98, top=0.92, bottom=0.10, wspace=0.25, hspace=0.30)
    p_grid = os.path.join(args.out_dir, "padim_panel_abcd_2x2.png")
    fig.savefig(p_grid, dpi=PNG_DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("Wrote", p_grid)


if __name__ == "__main__":
    main()
