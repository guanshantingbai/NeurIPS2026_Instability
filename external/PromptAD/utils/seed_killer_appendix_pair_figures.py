#!/usr/bin/env python3
"""
Generate additional risk–coverage figures for seed pairs (appendix material).

Excludes the primary killer pair from analysis/killer_pair.json, then picks the
next N pairs by mean |Δrisk| (with a small floor to skip degenerate curves).

Usage (from repo root):
  python PromptAD/utils/seed_killer_appendix_pair_figures.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROMPTAD_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _PROMPTAD_ROOT not in sys.path:
    sys.path.insert(0, _PROMPTAD_ROOT)

from utils.rejection_instability_analysis import (  # noqa: E402
    build_analysis_frame,
    load_and_validate_csv,
)
from utils.seed_killer_evidence_pipeline import (  # noqa: E402
    default_coverage_grid,
    find_per_sample_csv,
    risk_coverage,
)

PROXY_COL = "proxy_u6"


def _pair_key(slug: str, sa: int, sb: int) -> Tuple[str, int, int]:
    a, b = int(sa), int(sb)
    if a > b:
        a, b = b, a
    return (slug, a, b)


def load_killer_pair(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_one_pair(
    row: pd.Series,
    search_root: str,
    out_path: str,
    cov,
    log: logging.Logger,
) -> None:
    slug = str(row["slug"])
    ds = str(row["dataset"])
    cat = str(row["category"])
    k = int(row["k"])
    sa, sb = int(row["seed_a"]), int(row["seed_b"])

    root_a = os.path.join(search_root, slug, str(sa))
    root_b = os.path.join(search_root, slug, str(sb))
    csv_a = find_per_sample_csv(root_a, ds, k, sa, cat)
    csv_b = find_per_sample_csv(root_b, ds, k, sb, cat)
    if not csv_a or not csv_b:
        raise FileNotFoundError(f"Missing CSV for {slug} seeds {sa}/{sb}")

    dfa = build_analysis_frame(load_and_validate_csv(csv_a), log)
    dfb = build_analysis_frame(load_and_validate_csv(csv_b), log)
    ca = risk_coverage(dfa, PROXY_COL, cov)
    cb = risk_coverage(dfb, PROXY_COL, cov)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for c, lab in ((ca, f"seed {sa}"), (cb, f"seed {sb}")):
        xs = c["coverage"].to_numpy(dtype=float)
        ys = c["mean_error"].to_numpy(dtype=float)
        idx = np.argsort(xs)
        ax.plot(xs[idx], ys[idx], linewidth=2.0, label=lab)
    ax.set_xlabel("Coverage (fraction kept, low u6 first)")
    ax.set_ylabel("Risk (mean ranking error on kept set)")
    ax.set_title(f"Same setting, different seeds\n{slug.replace('__', ' / ')}")
    note = (
        f"AUROC: {float(row['auroc_a']):.4f} vs {float(row['auroc_b']):.4f}  "
        f"|Δ|={float(row['delta_auroc']):.5f}\n"
        f"I_mean: {float(row['instability_a']):.4f} vs {float(row['instability_b']):.4f}  "
        f"|Δ|={float(row['delta_instability']):.4f}\n"
        f"mean |Δrisk|: {float(row['mean_abs_risk_diff']):.4f}"
    )
    ax.text(0.02, 0.98, note, transform=ax.transAxes, va="top", fontsize=8, linespacing=1.15)
    ax.legend(frameon=False, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    log = logging.getLogger("appendix_figs")

    p = argparse.ArgumentParser()
    p.add_argument(
        "--search-root",
        type=str,
        default=os.path.join(_PROMPTAD_ROOT, "result_seed_search"),
    )
    p.add_argument(
        "--analysis-dir",
        type=str,
        default=os.path.join(_PROMPTAD_ROOT, "result_seed_search", "analysis"),
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=os.path.join(_PROMPTAD_ROOT, "result_seed_search", "appendix_extra_pairs"),
    )
    p.add_argument("--n-pairs", type=int, default=5)
    p.add_argument(
        "--min-mean-abs-risk",
        type=float,
        default=1e-5,
        help="Skip pairs with smaller mean |Δrisk| (numerical zeros).",
    )
    args = p.parse_args()

    search_root = os.path.abspath(args.search_root)
    cand_path = os.path.join(os.path.abspath(args.analysis_dir), "candidate_pairs_with_risk_metrics.csv")
    killer_path = os.path.join(os.path.abspath(args.analysis_dir), "killer_pair.json")
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(cand_path)
    df = df[df["error"].fillna("") == ""].copy()
    df = df[df["mean_abs_risk_diff"] >= float(args.min_mean_abs_risk)].copy()

    killer = load_killer_pair(killer_path)
    if killer:
        kkey = _pair_key(str(killer["slug"]), int(killer["seed_a"]), int(killer["seed_b"]))
        keys = df.apply(lambda r: _pair_key(str(r["slug"]), int(r["seed_a"]), int(r["seed_b"])), axis=1)
        df = df[keys != kkey].copy()

    df = df.sort_values("mean_abs_risk_diff", ascending=False, kind="mergesort").reset_index(drop=True)
    top = df.head(int(args.n_pairs))

    cov = default_coverage_grid()
    manifest: List[Dict[str, Any]] = []
    for i, (_, row) in enumerate(top.iterrows(), start=1):
        slug_short = str(row["slug"]).replace("__", "_")
        fname = f"appendix_pair{i:02d}_{slug_short}_seed{int(row['seed_a'])}_vs_seed{int(row['seed_b'])}.png"
        out_path = os.path.join(out_dir, fname)
        plot_one_pair(row, search_root, out_path, cov, log)
        manifest.append(
            {
                "file": fname,
                "slug": row["slug"],
                "seed_a": int(row["seed_a"]),
                "seed_b": int(row["seed_b"]),
                "delta_auroc": float(row["delta_auroc"]),
                "delta_instability": float(row["delta_instability"]),
                "mean_abs_risk_diff": float(row["mean_abs_risk_diff"]),
                "a_dominates_b": bool(row["a_dominates_b"]),
                "b_dominates_a": bool(row["b_dominates_a"]),
            }
        )
        print("Wrote", out_path)

    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"excluded_killer": killer, "figures": manifest}, f, indent=2, ensure_ascii=False)

    readme = os.path.join(out_dir, "README.md")
    lines = [
        "# Appendix: additional same-setting seed pairs",
        "",
        "Risk–coverage curves use proxy **u6(x)** (same definition as main killer analysis).",
        "The primary killer pair from `analysis/killer_pair.json` is **excluded** here.",
        "",
        "| # | File | Setting | Seeds | mean \\|Δrisk\\| | |ΔAUROC| | |ΔI| |",
        "|---|------|---------|-------|------------------|---------|------|",
    ]
    for idx, m in enumerate(manifest, start=1):
        lines.append(
            f"| {idx} | `{m['file']}` | `{m['slug']}` | {m['seed_a']} vs {m['seed_b']} | "
            f"{m['mean_abs_risk_diff']:.4f} | {m['delta_auroc']:.5f} | {m['delta_instability']:.4f} |"
        )
    with open(readme, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote {len(manifest)} figures under {out_dir}")


if __name__ == "__main__":
    main()
