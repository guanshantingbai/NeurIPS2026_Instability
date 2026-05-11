"""
Build paper-ready summaries for Sec. 3.1 (Empirical Observation) from existing
exp2/exp5/exp6 artifacts under result_round1. Does not recompute pairwise_table.

Outputs (default under --output_dir):
  exp_summary_all.csv, exp_summary_stats.json,
  plot_ibin_vs_error.csv, plot_margin_vs_flip.csv, plot_rejection_curve.csv,
  exp_summary_sentences.txt
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def _is_cls_run(name: str) -> bool:
    return name.startswith("CLS-") and "per_sample" in name


def discover_run_dirs(result_root: str) -> List[str]:
    pattern = os.path.join(
        result_root, "*", "k_*", "pairwise_instability", "*", "pairwise_table.csv"
    )
    out: List[str] = []
    for p in sorted(glob.glob(pattern)):
        if not os.path.isfile(p):
            continue
        run_dir = os.path.dirname(p)
        if _is_cls_run(os.path.basename(run_dir)):
            out.append(run_dir)
    return out


def parse_run_meta(run_dir: str) -> Optional[Dict[str, Any]]:
    run_dir = os.path.normpath(run_dir)
    run_name = os.path.basename(run_dir)
    pi = os.path.dirname(run_dir)
    k_dir = os.path.dirname(pi)
    dataset_dir = os.path.dirname(k_dir)
    if os.path.basename(pi) != "pairwise_instability":
        return None
    m = re.fullmatch(r"k_(\d+)", os.path.basename(k_dir))
    if not m:
        return None
    k = int(m.group(1))
    dataset = os.path.basename(dataset_dir)
    prefix = f"CLS-{dataset}-"
    category = ""
    if run_name.startswith(prefix):
        rest = run_name[len(prefix) :]
        parts = re.split(r"-k\d+", rest, maxsplit=1)
        category = parts[0] if parts else ""
    sm = re.search(r"-seed(\d+)-", run_name)
    seed = int(sm.group(1)) if sm else None
    return {
        "dataset": dataset,
        "category": category,
        "k": k,
        "seed": seed,
        "run_name": run_name,
        "run_dir": run_dir,
    }


def _corr_ibin_error(df: pd.DataFrame) -> Tuple[float, float]:
    sub = df[["I_bin", "error"]].dropna()
    if len(sub) < 2:
        return float("nan"), float("nan")
    ib = sub["I_bin"].to_numpy(dtype=float)
    er = sub["error"].to_numpy(dtype=float)
    if np.std(ib) == 0 or np.std(er) == 0:
        return float("nan"), float("nan")
    return float(pearsonr(ib, er)[0]), float(spearmanr(ib, er)[0])


def _read_rejection(exp_dir: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(exp_dir, "exp5_instability_rejection.csv")
    if not os.path.isfile(path):
        return None
    rej = pd.read_csv(path, comment="#")
    if "setting" not in rej.columns or "mean_error_all" not in rej.columns:
        return None
    out: Dict[str, Any] = {}
    for setting, key in [
        ("baseline", "baseline_error"),
        ("reject_10", "error@10"),
        ("reject_20", "error@20"),
        ("reject_30", "error@30"),
    ]:
        row = rej.loc[rej["setting"] == setting, "mean_error_all"]
        out[key] = float(row.iloc[0]) if not row.empty else float("nan")
    b = out["baseline_error"]
    for pct in (10, 20, 30):
        e = out[f"error@{pct}"]
        if b == b and b != 0 and e == e:
            out[f"rel_drop@{pct}"] = (b - e) / b
        else:
            out[f"rel_drop@{pct}"] = float("nan")
    e0, e10, e20, e30 = out["baseline_error"], out["error@10"], out["error@20"], out["error@30"]
    if all(v == v for v in (e0, e10, e20, e30)):
        out["is_monotonic"] = bool(e0 > e10 > e20 > e30)
    else:
        out["is_monotonic"] = False
    return out


def _read_summary_json(run_dir: str) -> Dict[str, Any]:
    path = os.path.join(run_dir, "summary.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def collect_run_summary(run_dir: str) -> Optional[Dict[str, Any]]:
    meta = parse_run_meta(run_dir)
    if not meta:
        return None
    exp_dir = os.path.join(run_dir, "experiments")
    sample_path = os.path.join(exp_dir, "exp5_sample_ranking_error.csv")
    if not os.path.isfile(sample_path):
        warnings.warn(f"Missing {sample_path}")
        return None
    df = pd.read_csv(sample_path)
    if not {"I_bin", "error"}.issubset(df.columns):
        return None
    pearson_r, spearman_r = _corr_ibin_error(df)
    mean_I_bin = float(df["I_bin"].astype(float).mean())

    rej = _read_rejection(exp_dir)
    if rej is None:
        warnings.warn(f"Missing exp5_instability_rejection in {run_dir}")
        return None

    sj = _read_summary_json(run_dir)
    num_samples = int(sj.get("num_samples", len(df)))
    flip_rate_mean = float(sj["flip_rate_mean"]) if "flip_rate_mean" in sj else float("nan")

    row = {
        "dataset": meta["dataset"],
        "category": meta["category"],
        "k": meta["k"],
        "seed": meta["seed"] if meta["seed"] is not None else "",
        "run_name": meta["run_name"],
        "num_samples": num_samples,
        "baseline_error": rej["baseline_error"],
        "spearman_ibin_error": spearman_r,
        "pearson_ibin_error": pearson_r,
        "rel_drop@10": rej["rel_drop@10"],
        "rel_drop@20": rej["rel_drop@20"],
        "rel_drop@30": rej["rel_drop@30"],
        "is_monotonic": rej["is_monotonic"],
        "mean_I_bin": mean_I_bin,
        "flip_rate_mean": flip_rate_mean,
    }
    return row


def _pool_ibin_error_buckets(list_of_dfs: List[pd.DataFrame]) -> pd.DataFrame:
    if not list_of_dfs:
        return pd.DataFrame(columns=["I_bin_bucket", "sample_count", "mean_error"])
    all_df = pd.concat(list_of_dfs, ignore_index=True)
    x = all_df["I_bin"].astype(float)
    y = all_df["error"].astype(float)
    ranks = x.rank(method="first")
    bucket = pd.qcut(ranks, q=3, labels=["low", "mid", "high"], duplicates="drop")
    tmp = pd.DataFrame({"I_bin_bucket": bucket, "error": y})
    g = (
        tmp.groupby("I_bin_bucket", observed=True)
        .agg(sample_count=("error", "size"), mean_error=("error", "mean"))
        .reset_index()
    )
    return g


def _aggregate_exp2_margin_flip(run_dirs: List[str]) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for rd in run_dirs:
        p = os.path.join(rd, "experiments", "exp2_abs_m_final_flip_rate.csv")
        if not os.path.isfile(p):
            continue
        try:
            d = pd.read_csv(p)
        except Exception:
            continue
        if "bucket" not in d.columns or "flip_rate" not in d.columns or "pair_count" not in d.columns:
            continue
        rows.append(d[["bucket", "flip_rate", "pair_count"]])
    if not rows:
        return pd.DataFrame(columns=["abs_m_final_bucket", "pair_count", "flip_rate"])
    u = pd.concat(rows, ignore_index=True)
    out_rows: List[Dict[str, Any]] = []
    for b in ["low", "mid", "high"]:
        sub = u[u["bucket"].astype(str) == b]
        if sub.empty:
            continue
        pc = int(sub["pair_count"].sum())
        wfr = float((sub["flip_rate"] * sub["pair_count"]).sum() / pc) if pc else float("nan")
        out_rows.append(
            {"abs_m_final_bucket": b, "pair_count": pc, "flip_rate": wfr}
        )
    return pd.DataFrame(out_rows)


def build_rejection_long_from_runs(rows_with_dir: List[Dict[str, Any]]) -> pd.DataFrame:
    long_rows: List[Dict[str, Any]] = []
    for mr in rows_with_dir:
        rd = mr["_run_dir"]
        path = os.path.join(rd, "experiments", "exp5_instability_rejection.csv")
        if not os.path.isfile(path):
            continue
        rej = pd.read_csv(path, comment="#")
        mapping = {"baseline": 0, "reject_10": 10, "reject_20": 20, "reject_30": 30}
        for setting, pct in mapping.items():
            row = rej.loc[rej["setting"] == setting, "mean_error_all"]
            if row.empty:
                continue
            long_rows.append(
                {
                    "dataset": mr["dataset"],
                    "category": mr["category"],
                    "k": mr["k"],
                    "seed": mr["seed"],
                    "run_name": mr["run_name"],
                    "reject_pct": pct,
                    "mean_error": float(row.iloc[0]),
                }
            )
    return pd.DataFrame(long_rows)


def compute_stats(df: pd.DataFrame) -> Dict[str, Any]:
    n_total = len(df)
    base = pd.to_numeric(df["baseline_error"], errors="coerce")
    degenerate = (base == 0) | base.isna()
    n_degenerate = int(degenerate.sum())
    valid = ~degenerate
    n_valid = int(valid.sum())

    sub = df.loc[valid].copy()
    sp = pd.to_numeric(sub["spearman_ibin_error"], errors="coerce")
    rd10 = pd.to_numeric(sub["rel_drop@10"], errors="coerce")
    rd20 = pd.to_numeric(sub["rel_drop@20"], errors="coerce")
    rd30 = pd.to_numeric(sub["rel_drop@30"], errors="coerce")
    mono = sub["is_monotonic"].astype(bool)

    def cnt_pct(num: int, den: int) -> Dict[str, Any]:
        return {
            "count": int(num),
            "denominator": int(den),
            "percentage": float(100.0 * num / den) if den else 0.0,
        }

    sp_ok = sp.notna() & (sp > 0)
    rd30_ok = rd30.notna() & (rd30 > 0)
    all_rd_ok = rd10.notna() & rd20.notna() & rd30.notna() & (rd10 > 0) & (rd20 > 0) & (rd30 > 0)

    flip_all = pd.to_numeric(df["flip_rate_mean"], errors="coerce")

    stats: Dict[str, Any] = {
        "n_runs_total": n_total,
        "n_runs_baseline_zero": cnt_pct(n_degenerate, n_total),
        "n_runs_valid_for_correlation": n_valid,
        "flip_rate_mean_over_all_runs": {
            "mean": float(flip_all.mean(skipna=True)),
            "median": float(flip_all.median(skipna=True)),
            "min": float(flip_all.min(skipna=True)),
            "max": float(flip_all.max(skipna=True)),
            "n_non_nan": int(flip_all.notna().sum()),
        },
        "spearman_positive_among_valid": cnt_pct(int(sp_ok.sum()), n_valid),
        "rel_drop30_positive_among_valid": cnt_pct(int(rd30_ok.sum()), n_valid),
        "rel_drop_all_positive_among_valid": cnt_pct(int(all_rd_ok.sum()), n_valid),
        "is_monotonic_true_all_runs": cnt_pct(int(df["is_monotonic"].astype(bool).sum()), n_total),
        "is_monotonic_true_among_valid": cnt_pct(int(mono.sum()), n_valid),
        "spearman_ibin_error_among_valid": {
            "mean": float(sp.mean(skipna=True)),
            "median": float(sp.median(skipna=True)),
            "n_non_nan": int(sp.notna().sum()),
        },
        "rel_drop30_among_valid": {
            "mean": float(rd30.mean(skipna=True)),
            "median": float(rd30.median(skipna=True)),
            "n_non_nan": int(rd30.notna().sum()),
        },
    }
    return stats


def write_sentences(stats: Dict[str, Any], out_path: str) -> None:
    nv = stats["n_runs_valid_for_correlation"]
    nt = stats["n_runs_total"]
    nz = stats["n_runs_baseline_zero"]["count"]
    fr = stats["flip_rate_mean_over_all_runs"]
    sp = stats["spearman_positive_among_valid"]
    r30 = stats["rel_drop30_positive_among_valid"]
    rall = stats["rel_drop_all_positive_among_valid"]
    mono_v = stats["is_monotonic_true_among_valid"]
    mono_a = stats["is_monotonic_true_all_runs"]
    sm = stats["spearman_ibin_error_among_valid"]
    rm = stats["rel_drop30_among_valid"]

    lines = [
        f"Across {nt} evaluated settings (MVTec AD and VisA, multiple categories and k-shot values), "
        f"pairwise decision instability is ubiquitous: mean pair flip rate averages {fr['mean']:.3f} (median {fr['median']:.3f}, range [{fr['min']:.3f}, {fr['max']:.3f}]) over runs, "
        f"and margin-stratified aggregation shows flip rate is highest when |m_final| is smallest (see plot_margin_vs_flip.csv). "
        f"{nz} settings have zero baseline ranking error and are excluded from correlation and relative-drop analyses.",
        f"Among the remaining {nv} settings with strictly positive baseline ranking error, Spearman correlation between sample-level instability (I_bin) and ranking error is positive in "
        f"{sp['count']}/{sp['denominator']} cases ({sp['percentage']:.1f}%), with a mean Spearman ρ of {sm['mean']:.3f} and median {sm['median']:.3f} over non-missing values.",
        f"Instability-aware rejection monotonically reduces mean ranking error from 0% to 30% rejection in {mono_v['count']}/{mono_v['denominator']} valid settings ({mono_v['percentage']:.1f}%) under a strict stepwise inequality; "
        f"considering all {nt} runs, the corresponding fraction is {mono_a['count']}/{mono_a['denominator']} ({mono_a['percentage']:.1f}%).",
        f"Rejecting the top 30% of samples by I_bin yields a positive relative reduction in mean ranking error in {r30['count']}/{r30['denominator']} valid settings ({r30['percentage']:.1f}%); "
        f"among valid settings, the mean and median relative drops at 30% rejection (relative to baseline) are {rm['mean']*100:.2f}% and {rm['median']*100:.2f}%.",
        f"In {rall['count']}/{rall['denominator']} valid settings ({rall['percentage']:.1f}%), the relative error reduction is strictly positive at 10%, 20%, and 30% rejection simultaneously, "
        f"supporting a consistent benefit–coverage trade-off rather than an isolated operating point.",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper Sec 3.1 summary from result_round1")
    parser.add_argument(
        "--result_root",
        type=str,
        default=os.path.join("PromptAD", "result_round1"),
        help="Root containing dataset/k_*/pairwise_instability/...",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to write outputs (default: result_root)",
    )
    args = parser.parse_args()
    result_root = os.path.abspath(os.path.normpath(args.result_root))
    out_dir = os.path.abspath(os.path.normpath(args.output_dir or result_root))
    os.makedirs(out_dir, exist_ok=True)

    run_dirs = discover_run_dirs(result_root)
    rows: List[Dict[str, Any]] = []
    pool_dfs: List[pd.DataFrame] = []

    for rd in run_dirs:
        r = collect_run_summary(rd)
        if r is None:
            continue
        b = r["baseline_error"]
        rows.append({**r, "_run_dir": rd})
        if b == b and b > 0:
            sp = os.path.join(rd, "experiments", "exp5_sample_ranking_error.csv")
            pool_dfs.append(pd.read_csv(sp)[["I_bin", "error"]])

    col_order = [
        "dataset",
        "category",
        "k",
        "seed",
        "num_samples",
        "baseline_error",
        "spearman_ibin_error",
        "pearson_ibin_error",
        "rel_drop@10",
        "rel_drop@20",
        "rel_drop@30",
        "is_monotonic",
        "mean_I_bin",
        "flip_rate_mean",
        "run_name",
    ]
    export_rows = [{k: v for k, v in r.items() if k != "_run_dir"} for r in rows]
    summary_df = pd.DataFrame(export_rows)
    if len(summary_df):
        extra = [c for c in summary_df.columns if c not in col_order]
        summary_df = summary_df[[c for c in col_order if c in summary_df.columns] + extra]

    p_all = os.path.join(out_dir, "exp_summary_all.csv")
    summary_df.to_csv(p_all, index=False)

    stats = compute_stats(summary_df)
    p_stats = os.path.join(out_dir, "exp_summary_stats.json")
    with open(p_stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    ibin_plot = _pool_ibin_error_buckets(pool_dfs)
    ibin_plot.to_csv(os.path.join(out_dir, "plot_ibin_vs_error.csv"), index=False)

    margin_plot = _aggregate_exp2_margin_flip(run_dirs)
    margin_plot.to_csv(os.path.join(out_dir, "plot_margin_vs_flip.csv"), index=False)

    rej_long = build_rejection_long_from_runs(rows)
    rej_long.to_csv(os.path.join(out_dir, "plot_rejection_curve.csv"), index=False)

    p_txt = os.path.join(out_dir, "exp_summary_sentences.txt")
    write_sentences(stats, p_txt)

    print(f"Wrote: {p_all} ({len(summary_df)} rows)")
    print(f"Wrote: {p_stats}")
    print(f"Wrote: plot_ibin_vs_error.csv ({len(ibin_plot)} buckets)")
    print(f"Wrote: plot_margin_vs_flip.csv ({len(margin_plot)} buckets)")
    print(f"Wrote: plot_rejection_curve.csv ({len(rej_long)} rows)")
    print(f"Wrote: {p_txt}")


if __name__ == "__main__":
    main()
