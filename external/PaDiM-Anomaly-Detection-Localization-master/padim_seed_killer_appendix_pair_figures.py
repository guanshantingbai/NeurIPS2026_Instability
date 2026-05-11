#!/usr/bin/env python3
"""
Appendix risk–coverage figures for PaDiM seed-killer pairs (excludes primary killer).

Reads ``analysis/candidate_pairs_with_risk_metrics.csv`` and ``analysis/killer_pair.json`` under
``padim_result_seed_search`` by default; writes PNGs under ``appendix_extra_pairs/``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_PADIM_ROOT = os.path.abspath(_HERE)
_REPO_ROOT = os.path.abspath(os.path.join(_PADIM_ROOT, ".."))
_PROMPTAD_ROOT = os.path.join(_REPO_ROOT, "external", "PromptAD")
if _PADIM_ROOT not in sys.path:
    sys.path.insert(0, _PADIM_ROOT)
if _PROMPTAD_ROOT not in sys.path:
    sys.path.insert(0, _PROMPTAD_ROOT)

from padim_seed_killer_evidence_pipeline import (  # noqa: E402
    PROXY_COL,
    build_analysis_frame_for_proxy,
    default_coverage_grid,
    find_per_sample_csv,
    risk_coverage,
)


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
    cov: np.ndarray,
    log: logging.Logger,
    proxy: str,
) -> None:
    slug = str(row["slug"])
    sa, sb = int(row["seed_a"]), int(row["seed_b"])
    proxy_col = PROXY_COL[proxy]

    root_a = os.path.join(search_root, slug, str(sa))
    root_b = os.path.join(search_root, slug, str(sb))
    csv_a = find_per_sample_csv(root_a)
    csv_b = find_per_sample_csv(root_b)
    if not csv_a or not csv_b:
        raise FileNotFoundError(f"Missing per_sample.csv for {slug} seeds {sa}/{sb}")

    dfa = build_analysis_frame_for_proxy(str(csv_a), proxy, log)
    dfb = build_analysis_frame_for_proxy(str(csv_b), proxy, log)
    ca = risk_coverage(dfa, proxy_col, cov)
    cb = risk_coverage(dfb, proxy_col, cov)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for c, lab in ((ca, f"seed {sa}"), (cb, f"seed {sb}")):
        xs = c["coverage"].to_numpy(dtype=float)
        ys = c["mean_error"].to_numpy(dtype=float)
        idx = np.argsort(xs)
        ax.plot(xs[idx], ys[idx], linewidth=2.0, label=lab)
    ax.set_xlabel(f"Coverage (fraction kept, low {proxy} first)")
    ax.set_ylabel("Risk (mean ranking error on kept set)")
    ax.set_title(f"PaDiM Protocol B — same setting, different seeds\n{slug.replace('__', ' / ')}")
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
    log = logging.getLogger("padim_appendix_figs")

    default_search = os.path.join(_PADIM_ROOT, "padim_result_seed_search")
    default_analysis = os.path.join(default_search, "analysis")

    p = argparse.ArgumentParser()
    p.add_argument("--search-root", type=str, default=default_search)
    p.add_argument("--analysis-dir", type=str, default=default_analysis)
    p.add_argument("--out-dir", type=str, default=os.path.join(default_search, "appendix_extra_pairs"))
    p.add_argument("--n-pairs", type=int, default=5)
    p.add_argument("--min-mean-abs-risk", type=float, default=1e-5)
    p.add_argument("--proxy", type=str, default="u6", choices=list(PROXY_COL.keys()))
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
        def _row_key(r: pd.Series) -> Tuple[str, int, int]:
            return _pair_key(str(r["slug"]), int(r["seed_a"]), int(r["seed_b"]))
        df["_pk"] = [_row_key(r) for _, r in df.iterrows()]
        df = df[df["_pk"] != kkey].drop(columns=["_pk"])

    df = df.sort_values("mean_abs_risk_diff", ascending=False, kind="mergesort").reset_index(drop=True)
    cov = default_coverage_grid()

    n = 0
    for i, row in df.iterrows():
        if n >= int(args.n_pairs):
            break
        slug = str(row["slug"])
        sa, sb = int(row["seed_a"]), int(row["seed_b"])
        out_png = os.path.join(out_dir, f"appendix_pair_{n+1:02d}__{slug}__{sa}__{sb}.png")
        try:
            plot_one_pair(row, search_root, out_png, cov, log, args.proxy)
            print(out_png)
            n += 1
        except Exception as e:
            print(f"skip row {i}: {e}")

    if n == 0:
        raise SystemExit("No appendix figures written (empty candidates or all failed).")


if __name__ == "__main__":
    main()
