"""
NeurIPS Figure 1: 2x2 PromptAD empirical analysis (Analysis_PromptAD_v2).

Panels:
  (a) AUROC vs instability  (b) Flip-rate distribution
  (c) Instability vs ranking error  (d) Rejection curve
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List, Sequence, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import image as mpimg

# Unified typography for standalone panel PNGs (composite uses these files as-is)
_PANEL_FS = 12
_PANEL_ANNOT_FS = 10
_TREND_COLOR = "#D98880"


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _load_auroc_for_row(result_root: str, row: pd.Series) -> float:
    p = os.path.join(
        result_root,
        str(row["dataset"]),
        f"k_{int(row['k'])}",
        "pairwise_instability",
        str(row["run_name"]),
        "summary.json",
    )
    if not os.path.isfile(p):
        return float("nan")
    try:
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        for key in ("sklearn_auroc_final", "pairwise_auroc_final", "auroc", "AUROC"):
            if key in d and d[key] is not None:
                return float(d[key])
    except Exception:
        pass
    return float("nan")


def _find_auroc_close_high_ibin_spread(
    auroc: np.ndarray, ibin: np.ndarray, max_auroc_gap: float = 0.03
) -> List[Tuple[int, int]]:
    """Return up to 2 disjoint pairs (i,j) with small AUROC gap and large I_bin spread."""
    n = len(auroc)
    pairs: List[Tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            da = abs(auroc[i] - auroc[j])
            if da <= max_auroc_gap:
                di = abs(ibin[i] - ibin[j])
                pairs.append((di, i, j))
    pairs.sort(key=lambda t: -t[0])
    used: Set[int] = set()
    out: List[Tuple[int, int]] = []
    for _, i, j in pairs:
        if i in used or j in used:
            continue
        out.append((i, j))
        used.update({i, j})
        if len(out) >= 2:
            break
    if not out and max_auroc_gap < 0.08:
        return _find_auroc_close_high_ibin_spread(auroc, ibin, max_auroc_gap=0.08)
    return out


def plot_panel_a_auroc_vs_instability(
    exp_summary_csv: str,
    result_root: str,
    out_path: str,
) -> None:
    df = pd.read_csv(exp_summary_csv)
    df = df.copy()
    df["AUROC"] = df.apply(lambda r: _load_auroc_for_row(result_root, r), axis=1)
    df = df.dropna(subset=["AUROC", "mean_I_bin"])
    df["AUROC"] = df["AUROC"].astype(float)
    df["mean_I_bin"] = df["mean_I_bin"].astype(float)

    x = df["AUROC"].to_numpy()
    y = df["mean_I_bin"].to_numpy()

    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    ax.scatter(x, y, alpha=0.45, s=28, edgecolors="none", color="0.35")

    # Annotate representative pairs: similar AUROC, different instability
    order = np.argsort(x)
    xo, yo = x[order], y[order]
    pairs_idx = _find_auroc_close_high_ibin_spread(xo, yo)
    ann_count = 0
    for ia, ib in pairs_idx:
        for idx in (int(order[ia]), int(order[ib])):
            r = df.iloc[idx]
            lab = f"{r['dataset'][:1].upper()}/{r['category'][:8]}"
            ax.annotate(
                lab,
                (float(r["AUROC"]), float(r["mean_I_bin"])),
                textcoords="offset points",
                xytext=(6, 6 if ann_count % 2 == 0 else -12),
                fontsize=_PANEL_ANNOT_FS,
                color="0.15",
                arrowprops=dict(arrowstyle="-", color="0.4", lw=0.6),
            )
            ann_count += 1
    if ann_count < 2 and len(df) >= 2:
        # fallback: label global max and min I_bin among mid AUROC band
        mid = (x.min() + x.max()) / 2
        band = df[(df["AUROC"] >= mid - 0.1) & (df["AUROC"] <= mid + 0.1)]
        if len(band) >= 2:
            hi = band.loc[band["mean_I_bin"].idxmax()]
            lo = band.loc[band["mean_I_bin"].idxmin()]
            for r in (hi, lo):
                lab = f"{r['dataset'][:1].upper()}/{r['category'][:8]}"
                ax.annotate(
                    lab,
                    (float(r["AUROC"]), float(r["mean_I_bin"])),
                    textcoords="offset points",
                    xytext=(6, 8),
                    fontsize=_PANEL_ANNOT_FS,
                    color="0.15",
                )

    ax.set_xlabel("AUROC", fontsize=_PANEL_FS)
    ax.set_ylabel("Instability score", fontsize=_PANEL_FS)
    ax.set_title("Similar AUROC, different instability", fontsize=_PANEL_FS + 1)
    ax.tick_params(axis="both", labelsize=_PANEL_FS)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def plot_panel_b_fliprate_distribution(exp_summary_csv: str, out_path: str) -> None:
    df = pd.read_csv(exp_summary_csv)
    vals = pd.to_numeric(df["flip_rate_mean"], errors="coerce").dropna().to_numpy()
    mean_v = float(np.mean(vals))
    max_v = float(np.max(vals))

    plt.rcParams.update(
        {
            "font.size": _PANEL_FS,
            "axes.labelsize": _PANEL_FS,
            "xtick.labelsize": _PANEL_FS,
            "ytick.labelsize": _PANEL_FS,
            "legend.fontsize": _PANEL_FS,
            "axes.linewidth": 0.8,
        }
    )
    fig, ax = plt.subplots(figsize=(5.0, 3.4), layout="constrained")
    xmax = min(1.0, max_v + 0.06)
    ax.hist(
        vals,
        bins=15,
        range=(0.0, xmax),
        color="0.45",
        edgecolor="0.15",
        linewidth=0.6,
    )
    ax.axvline(mean_v, color="red", linestyle="-", linewidth=1.2, label=f"mean flip rate = {mean_v:.2f}")
    ax.set_xlabel("Pairwise flip rate")
    ax.set_ylabel("Number of settings")
    ax.set_xlim(0, xmax)
    ax.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="0.5")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def plot_panel_c_instability_vs_error(sample_csv: str, out_path: str) -> None:
    df = pd.read_csv(sample_csv)
    x = df["I_bin"].astype(float)
    y = df["error"].astype(float)
    sp = float(x.corr(y, method="spearman"))

    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    ax.scatter(x, y, alpha=0.3, s=10, edgecolors="none", rasterized=True)

    mask = x.notna() & y.notna()
    xx = x[mask].to_numpy(dtype=float)
    yy = y[mask].to_numpy(dtype=float)
    if len(xx) >= 2 and float(np.std(xx)) > 1e-12:
        coef = np.polyfit(xx, yy, 1)
        xs = np.linspace(float(np.min(xx)), float(np.max(xx)), 100)
        ax.plot(xs, np.polyval(coef, xs), color=_TREND_COLOR, linewidth=2.0, zorder=3)

    if sp == sp:
        ax.text(
            0.98,
            0.98,
            rf"Spearman $\rho$ = {sp:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=_PANEL_FS,
            color=_TREND_COLOR,
        )

    ax.set_xlabel("Instability score", fontsize=_PANEL_FS)
    ax.set_ylabel("Ranking error", fontsize=_PANEL_FS)
    ax.tick_params(axis="both", labelsize=_PANEL_FS)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def plot_panel_d_rejection_curve(rejection_csv: str, out_path: str) -> None:
    rej = pd.read_csv(rejection_csv, comment="#")
    mapping = {"baseline": 0.0, "reject_10": 10.0, "reject_20": 20.0, "reject_30": 30.0}
    rows = []
    for s, pct in mapping.items():
        row = rej.loc[rej["setting"] == s, "mean_error_all"]
        if not row.empty:
            rows.append((pct, float(row.iloc[0])))
    rows.sort(key=lambda t: t[0])
    xs = np.array([t[0] for t in rows])
    ys = np.array([t[1] for t in rows])

    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.plot(xs, ys, marker="o", linewidth=1.5, color="0.2")
    ax.set_xlabel("Rejection rate (%)", fontsize=_PANEL_FS)
    ax.set_ylabel("Mean ranking error", fontsize=_PANEL_FS)
    ax.set_title("Error reduction by instability-aware rejection", fontsize=_PANEL_FS + 1)
    ax.tick_params(axis="both", labelsize=_PANEL_FS)
    ax.set_xticks(xs)
    for px, py in zip(xs, ys):
        ax.annotate(
            f"{py:.4f}",
            (px, py),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=_PANEL_ANNOT_FS,
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def build_composite_2x2(
    paths: Sequence[str],
    subtitles: Sequence[str],
    out_png: str,
    out_pdf: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.8))
    axes_flat = axes.flatten()
    subtitle_fs = 10
    panel_fs = 12

    for ax, p, sub in zip(axes_flat, paths, subtitles):
        if not os.path.isfile(p):
            raise FileNotFoundError(p)
        img = mpimg.imread(p)
        ax.imshow(img)
        ax.axis("off")
        ax.text(
            0.5,
            -0.06,
            sub,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=subtitle_fs,
            color="0.2",
        )

    for ax, letter in zip(axes_flat, ("a", "b", "c", "d")):
        ax.text(
            0.02,
            0.98,
            f"({letter})",
            transform=ax.transAxes,
            fontsize=panel_fs,
            fontweight="bold",
            va="top",
            ha="left",
            color="0.1",
        )

    plt.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.06, wspace=0.08, hspace=0.22)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def write_caption(out_path: str) -> None:
    text = """Figure (PromptAD empirical analysis). We summarize behavior across PromptAD-style settings on MVTec AD and VisA.
(a) Image-level AUROC alone leaves instability largely unresolved: settings with comparable AUROC can exhibit markedly different instability scores, indicating a blind spot for reliability assessment.
(b) Pairwise decision instability, measured by mean flip rate, is pervasive rather than confined to isolated failures; the distribution over settings concentrates away from zero.
(c) Instability aligns with ranking error: higher instability scores correspond to larger per-sample ranking error, with a strongly positive Spearman correlation in representative runs.
(d) Instability-aware sample rejection monotonically lowers mean ranking error as more unstable samples are withheld, showing that instability is actionable for improving decision reliability.
"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text.strip() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_root",
        type=str,
        default=os.path.join("PromptAD", "result_round1"),
    )
    parser.add_argument(
        "--exp_summary_csv",
        type=str,
        default=None,
        help="Default: {result_root}/exp_summary_all.csv",
    )
    parser.add_argument(
        "--macaroni_exp_dir",
        type=str,
        default=(
            "PromptAD/result_round1/visa/k_2/pairwise_instability/"
            "CLS-visa-macaroni1-k2-seed111-per_sample/experiments"
        ),
    )
    args = parser.parse_args()

    root = os.path.abspath(os.path.join(_repo_root(), args.result_root))
    exp_csv = os.path.abspath(
        os.path.join(_repo_root(), args.exp_summary_csv or os.path.join(args.result_root, "exp_summary_all.csv"))
    )
    mac_dir = os.path.abspath(os.path.join(_repo_root(), args.macaroni_exp_dir))
    fig_dir = os.path.join(root, "figures")

    p_a = os.path.join(fig_dir, "auroc_vs_instability.png")
    p_b = os.path.join(fig_dir, "fliprate_distribution.png")
    p_c = os.path.join(fig_dir, "instability_vs_error.png")
    p_d = os.path.join(fig_dir, "rejection_curve.png")
    out_png = os.path.join(fig_dir, "Analysis_PromptAD_v2.png")
    out_pdf = os.path.join(fig_dir, "Analysis_PromptAD_v2.pdf")
    cap_path = os.path.join(fig_dir, "Analysis_PromptAD_v2_caption.txt")

    plot_panel_a_auroc_vs_instability(exp_csv, root, p_a)
    plot_panel_b_fliprate_distribution(exp_csv, p_b)
    plot_panel_c_instability_vs_error(os.path.join(mac_dir, "exp5_sample_ranking_error.csv"), p_c)
    plot_panel_d_rejection_curve(os.path.join(mac_dir, "exp5_instability_rejection.csv"), p_d)

    subtitles = (
        "Similar AUROC, different instability",
        "Instability is pervasive",
        "Instability concentrates in error-prone regions",
        "Error reduction by instability-aware rejection",
    )
    build_composite_2x2((p_a, p_b, p_c, p_d), subtitles, out_png, out_pdf)
    write_caption(cap_path)

    print("Wrote:", p_a)
    print("Wrote:", p_b)
    print("Wrote:", p_c)
    print("Wrote:", p_d)
    print("Wrote:", out_png)
    print("Wrote:", out_pdf)
    print("Wrote:", cap_path)


if __name__ == "__main__":
    main()
