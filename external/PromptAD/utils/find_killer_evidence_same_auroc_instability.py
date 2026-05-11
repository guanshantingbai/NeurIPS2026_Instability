#!/usr/bin/env python3
"""
Search PromptAD result_round1 for a "killer" pair of settings (same dataset × class × k)
with nearly identical AUROC but very different run-level instability, and different
risk–coverage behavior when ranking/rejecting by a per-sample instability proxy u(x).

Steps (paper-oriented):
  1) Sort settings by AUROC inside a pool (--pool global default). Each result
     zip in this repo is one row per (dataset, class, k), so there is no second
     setting *within* the same cell; use ``global`` (81 runs) or ``same_dataset_k``
     (several classes at fixed k). Link rank neighbors at offsets ±window
     (default 3). Keep pairs with |ΔAUROC| < tau (default 0.01; prefer tighter
     0.003 when ranking killers). Rank pairs by |ΔI|. Write Top-K pairs.
  2) For each candidate, load per-sample table (precomputed analysis CSV if present,
     else build from raw *per_sample.csv), compute risk–coverage using the same
     rule as rejection_instability_analysis (keep low-proxy prefix; risk = mean
     sample_error on kept set).
  3) Pick one killer pair: prefer smallest ΔAUROC, then largest instability gap,
     then largest mean |Δrisk(c)| on the shared coverage grid; prefer strict
     dominance on mean_error if present.
  4) Save killer risk–coverage figure (x=coverage, y=risk) with paper title.

Usage (from repo root NeurIPS2026):
  python PromptAD/utils/find_killer_evidence_same_auroc_instability.py \\
    --result-root PromptAD/result_round1

  python PromptAD/utils/find_killer_evidence_same_auroc_instability.py \\
    --result-root PromptAD/result_round1 --proxy u6 --top-pairs 10
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Allow `python PromptAD/utils/this_script.py` from repo root
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROMPTAD_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _PROMPTAD_ROOT not in sys.path:
    sys.path.insert(0, _PROMPTAD_ROOT)

from utils.rejection_instability_analysis import (  # noqa: E402
    build_analysis_frame,
    load_and_validate_csv,
    _deterministic_rejection_curve,
)

LOG = logging.getLogger("killer_evidence")

PROXY_COL = {
    "u1": "proxy_instability",
    "u2": "proxy_u2",
    "u3": "proxy_u3",
    "u4": "proxy_u4",
    "u5": "proxy_u5",
    "u6": "proxy_u6",
}


def _repo_root() -> str:
    return os.path.abspath(os.path.join(_HERE, "..", ".."))


def _parse_k(folder_name: str) -> Optional[int]:
    if folder_name.startswith("k_"):
        try:
            return int(folder_name.split("_", 1)[1])
        except ValueError:
            return None
    return None


def parse_category_from_run(run_name: str, dataset: str, k: int) -> str:
    prefix = f"CLS-{dataset}-"
    suffix = f"-k{k}-"
    if run_name.startswith(prefix) and suffix in run_name:
        return run_name[len(prefix) : run_name.index(suffix)]
    return run_name


def load_summary(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def auroc_from_summary(d: Dict[str, Any]) -> float:
    for key in ("sklearn_auroc_final", "pairwise_auroc_final", "auroc", "AUROC"):
        if key in d and d[key] is not None:
            return float(d[key])
    return float("nan")


def instability_from_summary(d: Dict[str, Any]) -> float:
    for key in ("mean_I_bin", "flip_rate_mean", "anomaly_I_bin_mean"):
        if key in d and d[key] is not None:
            return float(d[key])
    return float("nan")


def discover_pairwise_runs(result_root: str) -> List[Dict[str, Any]]:
    import glob

    pattern = os.path.join(result_root, "*", "k_*", "pairwise_instability", "*", "summary.json")
    runs: List[Dict[str, Any]] = []
    for summary_path in sorted(glob.glob(pattern)):
        run_dir = os.path.dirname(summary_path)
        rel = os.path.relpath(summary_path, result_root)
        parts = rel.split(os.sep)
        if len(parts) < 4:
            continue
        dataset = parts[0]
        k = _parse_k(parts[1])
        if k is None:
            continue
        run_name = parts[3]
        # Ignore stray folders (e.g. duplicated summaries under short names).
        if not str(run_name).startswith("CLS-"):
            continue
        category = parse_category_from_run(run_name, dataset, k)
        runs.append(
            {
                "dataset": dataset,
                "category": category,
                "k": k,
                "run_name": run_name,
                "run_dir": run_dir,
                "summary_path": summary_path,
            }
        )
    return runs


def enrich_run(r: Dict[str, Any]) -> Dict[str, Any]:
    d = load_summary(r["summary_path"])
    o = dict(r)
    o["auroc"] = auroc_from_summary(d)
    o["instability"] = instability_from_summary(d)
    return o


def precomputed_analysis_path(result_root: str, dataset: str, k: int, run_name: str) -> str:
    return os.path.join(result_root, dataset, f"k_{int(k)}", "rejection_instability", run_name, "per_sample_instability_analysis.csv")


def raw_per_sample_csv_path(result_root: str, dataset: str, k: int, run_name: str) -> str:
    return os.path.join(result_root, dataset, f"k_{int(k)}", "csv", f"{run_name}.csv")


def load_analyzed_frame(result_root: str, row: Dict[str, Any], log: logging.Logger) -> Optional[pd.DataFrame]:
    """Return frame with columns required for risk–coverage (incl. sample_error, image_label, harmonic_score)."""
    pc = precomputed_analysis_path(result_root, row["dataset"], int(row["k"]), row["run_name"])
    if os.path.isfile(pc):
        df = pd.read_csv(pc)
        need = ["image_label", "harmonic_score", "sample_error"] + [PROXY_COL[k] for k in PROXY_COL]
        miss = [c for c in need if c not in df.columns]
        if miss:
            log.warning("Precomputed %s missing columns %s; trying raw CSV", pc, miss)
        else:
            return df

    raw = raw_per_sample_csv_path(result_root, row["dataset"], int(row["k"]), row["run_name"])
    if not os.path.isfile(raw):
        log.warning("No per-sample data for %s / %s", row["run_name"], raw)
        return None
    base = load_and_validate_csv(raw)
    return build_analysis_frame(base, log)


def default_coverage_grid() -> np.ndarray:
    return np.clip(np.round(np.arange(1.0, 0.45, -0.05), 2), 0.5, 1.0)


def risk_coverage_instability_proxy(df: pd.DataFrame, proxy_col: str, coverage_grid: np.ndarray) -> pd.DataFrame:
    """Same ordering as run_rejection_experiment: ascending proxy, NaN -> +inf (dropped first)."""
    u = df[proxy_col].to_numpy(dtype=float)
    u_sort = np.where(np.isfinite(u), u, np.inf)
    order = np.argsort(u_sort, kind="mergesort")
    return _deterministic_rejection_curve(df, order, coverage_grid)


def trapz_y_x(x: np.ndarray, y: np.ndarray) -> float:
    """∫ y dx with x sorted (any direction)."""
    order = np.argsort(x)
    xs = x[order]
    ys = y[order]
    # numpy<2.0 uses trapz
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(ys, xs))
    return float(np.trapz(ys, xs))


def curve_metrics(
    curve_a: pd.DataFrame, curve_b: pd.DataFrame
) -> Dict[str, Any]:
    m = curve_a.merge(curve_b, on="coverage", suffixes=("_a", "_b"))
    ra = m["mean_error_a"].to_numpy(dtype=float)
    rb = m["mean_error_b"].to_numpy(dtype=float)
    cov = m["coverage"].to_numpy(dtype=float)
    diff = ra - rb
    mean_abs = float(np.mean(np.abs(diff)))
    mean_signed = float(np.mean(diff))
    # Dominance: A better iff lower risk everywhere
    a_dom = bool(np.all(ra < rb - 1e-15))
    b_dom = bool(np.all(rb < ra - 1e-15))
    aurc_a = trapz_y_x(cov, ra)
    aurc_b = trapz_y_x(cov, rb)
    return {
        "mean_abs_risk_diff": mean_abs,
        "mean_signed_risk_a_minus_b": mean_signed,
        "a_dominates_b_on_risk": a_dom,
        "b_dominates_a_on_risk": b_dom,
        "aurc_a": aurc_a,
        "aurc_b": aurc_b,
        "aurc_abs_diff": abs(aurc_a - aurc_b),
    }


def build_neighbor_pairs(
    df: pd.DataFrame,
    neighbor_window: int,
    delta_auroc_max: float,
) -> pd.DataFrame:
    """
    Build undirected pairs from one AUROC-sorted pool (e.g. all settings globally,
    or all runs sharing (dataset, k)). Neighbors are ±neighbor_window positions
    in the sorted list (by AUROC, then run_name).
    """
    sub = df.dropna(subset=["auroc", "instability"]).copy()
    sub = sub.sort_values(["auroc", "run_name"], kind="mergesort").reset_index(drop=True)
    n = len(sub)
    rows: List[Dict[str, Any]] = []
    for i in range(n):
        for off in range(-neighbor_window, neighbor_window + 1):
            if off == 0:
                continue
            j = i + off
            if j < 0 or j >= n:
                continue
            a = sub.iloc[i]
            b = sub.iloc[j]
            d_auroc = abs(float(a["auroc"]) - float(b["auroc"]))
            d_inst = abs(float(a["instability"]) - float(b["instability"]))
            if d_auroc >= delta_auroc_max:
                continue
            if d_inst <= 0.0:
                continue
            if min(float(a["instability"]), float(b["instability"])) < 1e-8:
                continue
            r1, r2 = sorted([str(a["run_name"]), str(b["run_name"])])
            rows.append(
                {
                    "dataset_a": a["dataset"],
                    "dataset_b": b["dataset"],
                    "category_a": a["category"],
                    "category_b": b["category"],
                    "k_a": int(a["k"]),
                    "k_b": int(b["k"]),
                    "run_a": a["run_name"],
                    "run_b": b["run_name"],
                    "auroc_a": float(a["auroc"]),
                    "auroc_b": float(b["auroc"]),
                    "instability_a": float(a["instability"]),
                    "instability_b": float(b["instability"]),
                    "delta_auroc": d_auroc,
                    "delta_instability": d_inst,
                    "pair_key": f"{r1}||{r2}",
                }
            )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out = out.drop_duplicates(subset=["pair_key"])
    out = out.sort_values("delta_instability", ascending=False, kind="mergesort")
    return out


def build_pairs_for_pool_mode(
    meta: pd.DataFrame,
    pool: str,
    neighbor_window: int,
    delta_auroc_max: float,
) -> pd.DataFrame:
    pool = pool.strip().lower()
    if pool == "global":
        return build_neighbor_pairs(meta, neighbor_window, delta_auroc_max)
    if pool == "same_dataset_k":
        chunks: List[pd.DataFrame] = []
        for (_, _), g in meta.groupby(["dataset", "k"], sort=False):
            part = build_neighbor_pairs(g, neighbor_window, delta_auroc_max)
            if not part.empty:
                chunks.append(part)
        if not chunks:
            return pd.DataFrame()
        out = pd.concat(chunks, ignore_index=True)
        out = out.drop_duplicates(subset=["pair_key"])
        out = out.sort_values("delta_instability", ascending=False, kind="mergesort")
        return out
    raise ValueError(f"Unknown --pool {pool!r} (use global or same_dataset_k)")


def row_from_runname(meta: pd.DataFrame, run_name: str) -> Optional[pd.Series]:
    m = meta[meta["run_name"] == run_name]
    if m.empty:
        return None
    return m.iloc[0]


def plot_killer_rcurve(
    curve_a: pd.DataFrame,
    curve_b: pd.DataFrame,
    label_a: str,
    label_b: str,
    out_path: str,
    title: str = "Same AUROC, different decision reliability",
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    # Coverage increasing left → right for paper readability
    for c, lab, sty in (
        (curve_a, label_a, "-"),
        (curve_b, label_b, "--"),
    ):
        xs = c["coverage"].to_numpy(dtype=float)
        ys = c["mean_error"].to_numpy(dtype=float)
        idx = np.argsort(xs)
        ax.plot(xs[idx], ys[idx], sty, linewidth=2.0, label=lab)
    ax.set_xlabel("Coverage (fraction of samples kept, low proxy first)")
    ax.set_ylabel("Risk (mean ranking error on kept set)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def select_killer_row(cands: pd.DataFrame, shortlist: int = 8) -> Optional[pd.Series]:
    """
    Restrict to the top ``shortlist`` pairs by mean |Δrisk| (curve separation), then pick
    by same-dataset contrast, tight AUROC, smaller ΔAUROC, larger ΔI, dominance.
    """
    if cands.empty:
        return None
    s = cands.copy()
    s["_tight_auroc"] = s["delta_auroc"] < 0.003
    s["_dom"] = s["a_dominates_b_on_risk"] | s["b_dominates_a_on_risk"]
    if "dataset_a" in s.columns and "dataset_b" in s.columns:
        s["_same_ds"] = s["dataset_a"] == s["dataset_b"]
    else:
        s["_same_ds"] = False
    n = max(1, int(shortlist))
    tier = s.nlargest(n, "mean_abs_risk_diff")
    tier = tier.sort_values(
        by=[
            "_same_ds",
            "_tight_auroc",
            "delta_auroc",
            "delta_instability",
            "mean_abs_risk_diff",
            "_dom",
        ],
        ascending=[False, False, True, False, False, False],
        kind="mergesort",
    )
    return tier.iloc[0]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--result-root",
        type=str,
        default="PromptAD/result_round1",
        help="Path to result_round1 (absolute or relative to repo root).",
    )
    p.add_argument("--neighbor-window", type=int, default=3)
    p.add_argument("--delta-auroc-max", type=float, default=0.01)
    p.add_argument(
        "--pool",
        type=str,
        default="global",
        choices=("global", "same_dataset_k"),
        help="AUROC-rank neighbor pool: all 81 settings, or stratify by (dataset, k).",
    )
    p.add_argument("--proxy", type=str, default="u6", choices=sorted(PROXY_COL.keys()))
    p.add_argument(
        "--top-pairs",
        type=int,
        default=10,
        help="Write this many rows to top*_candidate_pairs.csv (ranked by Δinstability).",
    )
    p.add_argument(
        "--killer-eval-budget",
        type=int,
        default=50,
        help="Evaluate risk–coverage for the first N pairs in that ranking (should be >= top-pairs).",
    )
    p.add_argument(
        "--killer-shortlist",
        type=int,
        default=8,
        help="Final killer is chosen from this many pairs with largest mean |Δrisk| among evaluated candidates.",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Default: <result-root>/killer_evidence_search",
    )
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    root = os.path.abspath(os.path.join(_repo_root(), args.result_root))
    out_dir = args.out_dir.strip() or os.path.join(root, "killer_evidence_search")
    os.makedirs(out_dir, exist_ok=True)

    runs = discover_pairwise_runs(root)
    if not runs:
        raise SystemExit(f"No pairwise_instability runs under {root}")

    meta = pd.DataFrame([enrich_run(r) for r in runs])
    meta_path = os.path.join(out_dir, "all_settings_auroc_instability.csv")
    meta.to_csv(meta_path, index=False)
    LOG.info("Wrote %s (%d settings)", meta_path, len(meta))

    pairs_df = build_pairs_for_pool_mode(meta, args.pool, args.neighbor_window, args.delta_auroc_max)
    if pairs_df.empty:
        raise SystemExit(
            "No neighbor pairs satisfy |ΔAUROC| < delta-auroc-max and Δinstability>0. "
            "Try --pool global, increase --neighbor-window, or relax --delta-auroc-max."
        )
    pairs_df = pairs_df.sort_values(["delta_instability", "delta_auroc"], ascending=[False, True], kind="mergesort")
    top_write = pairs_df.head(int(args.top_pairs)).copy()
    top_path = os.path.join(out_dir, f"top{args.top_pairs}_candidate_pairs.csv")
    top_write.to_csv(top_path, index=False)
    LOG.info("Wrote %s", top_path)

    eval_budget = max(int(args.top_pairs), int(args.killer_eval_budget))
    top_eval = pairs_df.head(eval_budget).copy()

    proxy_col = PROXY_COL[args.proxy]
    cov = default_coverage_grid()

    eval_rows: List[Dict[str, Any]] = []
    mini_log = logging.getLogger("killer_evidence.load")

    for _, pr in top_eval.iterrows():
        ra = row_from_runname(meta, pr["run_a"])
        rb = row_from_runname(meta, pr["run_b"])
        if ra is None or rb is None:
            continue
        dfa = load_analyzed_frame(root, ra.to_dict(), mini_log)
        dfb = load_analyzed_frame(root, rb.to_dict(), mini_log)
        if dfa is None or dfb is None:
            eval_rows.append({**pr.to_dict(), "error": "missing_per_sample_data"})
            continue
        ca = risk_coverage_instability_proxy(dfa, proxy_col, cov)
        cb = risk_coverage_instability_proxy(dfb, proxy_col, cov)
        cm = curve_metrics(ca, cb)
        eval_rows.append({**pr.to_dict(), **cm, "error": ""})

    eval_df = pd.DataFrame(eval_rows)
    eval_path = os.path.join(out_dir, "candidate_pairs_risk_coverage_metrics.csv")
    eval_df.to_csv(eval_path, index=False)
    LOG.info("Wrote %s", eval_path)

    ok = eval_df[eval_df["error"] == ""].copy()
    if ok.empty:
        raise SystemExit("No candidate pair had loadable per-sample tables; run batch_rejection_instability_all_settings or ensure csv/ exists.")

    killer_s = select_killer_row(ok, shortlist=int(args.killer_shortlist))
    if killer_s is None:
        raise SystemExit("Internal error: empty ok frame.")
    killer_all = killer_s.to_dict()
    killer = {k: v for k, v in killer_all.items() if not str(k).startswith("_")}
    killer_path = os.path.join(out_dir, "killer_pair_choice.json")
    with open(killer_path, "w", encoding="utf-8") as f:
        json.dump(killer, f, indent=2, ensure_ascii=False)
    LOG.info("Wrote %s", killer_path)

    ra = row_from_runname(meta, killer["run_a"])
    rb = row_from_runname(meta, killer["run_b"])
    assert ra is not None and rb is not None
    dfa = load_analyzed_frame(root, ra.to_dict(), mini_log)
    dfb = load_analyzed_frame(root, rb.to_dict(), mini_log)
    assert dfa is not None and dfb is not None
    ca = risk_coverage_instability_proxy(dfa, proxy_col, cov)
    cb = risk_coverage_instability_proxy(dfb, proxy_col, cov)

    short_a = str(killer["run_a"]).replace("-per_sample", "")
    short_b = str(killer["run_b"]).replace("-per_sample", "")
    fig_png = os.path.join(out_dir, "killer_risk_coverage_same_auroc.png")
    plot_killer_rcurve(ca, cb, f"A: {short_a}", f"B: {short_b}", fig_png)

    dom_note = ""
    if killer.get("b_dominates_a_on_risk") and not killer.get("a_dominates_b_on_risk"):
        dom_note = " Setting B uniformly achieves lower risk across the coverage grid."
    elif killer.get("a_dominates_b_on_risk") and not killer.get("b_dominates_a_on_risk"):
        dom_note = " Setting A uniformly achieves lower risk across the coverage grid."
    caption = (
        f"Two PromptAD settings (A: {killer['dataset_a']}/{killer['category_a']}/k={killer['k_a']}; "
        f"B: {killer['dataset_b']}/{killer['category_b']}/k={killer['k_b']}) "
        f"have nearly identical image-level AUROC "
        f"({killer['auroc_a']:.4f} vs {killer['auroc_b']:.4f}, |Δ|={killer['delta_auroc']:.4f}) "
        f"but very different mean pairwise instability "
        f"({killer['instability_a']:.4f} vs {killer['instability_b']:.4f}, |Δ|={killer['delta_instability']:.4f}). "
        f"When each setting ranks its own test samples by proxy {args.proxy}(x) and we vary coverage, "
        f"the risk–coverage curves separate (mean |Δrisk| on the shared grid = {killer['mean_abs_risk_diff']:.4f})."
        f"{dom_note} "
        f"Thus matched AUROC does not imply matched decision reliability."
    )
    cap_path = os.path.join(out_dir, "killer_pair_paper_blurb.txt")
    with open(cap_path, "w", encoding="utf-8") as f:
        f.write(caption + "\n")
    LOG.info("Wrote %s", cap_path)
    LOG.info("Figure: %s", fig_png)
    print(caption)


if __name__ == "__main__":
    main()
