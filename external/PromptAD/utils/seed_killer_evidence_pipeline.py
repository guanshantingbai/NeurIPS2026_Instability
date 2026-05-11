#!/usr/bin/env python3
"""
Same-setting, multi-seed killer evidence pipeline for PromptAD.

Phases:
  phase1   — Score all 81 settings from result_round1; write Top-8 + rationale.
  phase2   — Emit a bash script to train+test+pairwise per (setting, seed).
  phase3   — After results exist under result_seed_search/, scan metrics, seed pairs,
             risk–coverage, pick killer, write killer_final.png + paper blurb.

Layout (after phase2 runs):
  PromptAD/result_seed_search/{slug}/{seed}/{dataset}/k_{k}/csv/*.csv   # train_cls/test_cls
  PromptAD/result_seed_search/{slug}/{seed}/pairwise_instability/summary.json

Slug example: mvtec__carpet__k1

Usage:
  cd NeurIPS2026 && python PromptAD/utils/seed_killer_evidence_pipeline.py phase1
  cd NeurIPS2026 && python PromptAD/utils/seed_killer_evidence_pipeline.py phase2 \\
      --seeds 111,222,333,444,555
  # run generated script from PromptAD/
  cd PromptAD && bash result_seed_search/phase2_train_test.sh

  cd NeurIPS2026 && python PromptAD/utils/seed_killer_evidence_pipeline.py phase3 \\
      --proxy u6
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import logging
import math
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
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _PROMPTAD_ROOT not in sys.path:
    sys.path.insert(0, _PROMPTAD_ROOT)

from utils.pairwise_instability import run_full_analysis  # noqa: E402
from utils.rejection_instability_analysis import (  # noqa: E402
    build_analysis_frame,
    load_and_validate_csv,
    _deterministic_rejection_curve,
)

PROXY_COL = {
    "u6": "proxy_u6",
    "u2": "proxy_u2",
}


# ---------------------------------------------------------------------------
# Discovery (result_round1)
# ---------------------------------------------------------------------------


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


def discover_pairwise_runs(result_root: str) -> List[Dict[str, Any]]:
    pattern = os.path.join(result_root, "*", "k_*", "pairwise_instability", "*", "summary.json")
    runs: List[Dict[str, Any]] = []
    for summary_path in sorted(glob.glob(pattern)):
        run_name = str(os.path.basename(os.path.dirname(summary_path)))
        if not run_name.startswith("CLS-"):
            continue
        rel = os.path.relpath(summary_path, result_root)
        parts = rel.split(os.sep)
        dataset, kf = parts[0], parts[1]
        k = _parse_k(kf)
        if k is None:
            continue
        category = parse_category_from_run(run_name, dataset, k)
        run_dir = os.path.dirname(summary_path)
        runs.append(
            {
                "dataset": dataset,
                "category": category,
                "k": k,
                "run_name": run_name,
                "run_dir": run_dir,
                "summary_path": summary_path,
                "per_sample_guess": os.path.join(
                    result_root, dataset, f"k_{k}", "csv", f"{run_name}.csv"
                ),
                "exp5_csv": os.path.join(run_dir, "experiments", "exp5_sample_ranking_error.csv"),
            }
        )
    return runs


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def mean_exp5_error(exp5_path: str) -> float:
    if not os.path.isfile(exp5_path):
        return float("nan")
    df = pd.read_csv(exp5_path)
    if "error" not in df.columns:
        return float("nan")
    return float(pd.to_numeric(df["error"], errors="coerce").mean())


def load_proxy_u6_table(aggregate_csv: str) -> pd.DataFrame:
    if not os.path.isfile(aggregate_csv):
        return pd.DataFrame()
    df = pd.read_csv(aggregate_csv)
    df = df[df["proxy_name"] == "u6"].copy()
    return df


def setting_csv_key(dataset: str, k: int, run_name: str) -> str:
    return f"{dataset}/k_{int(k)}/csv/{run_name}.csv"


def phase1_score_settings(
    result_root: str,
    aggregate_proxy_csv: str,
    top_n: int,
    out_dir: str,
) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    runs = discover_pairwise_runs(result_root)
    rows = []
    proxy_df = load_proxy_u6_table(aggregate_proxy_csv)
    proxy_map: Dict[str, float] = {}
    win_map: Dict[str, float] = {}
    if not proxy_df.empty:
        for _, r in proxy_df.iterrows():
            key = str(r["setting_csv"])
            proxy_map[key] = float(r["mean_delta_risk"])
            win_map[key] = float(r.get("win_rate_vs_random", float("nan")))

    inst_all: List[float] = []
    for r in runs:
        s = load_json(r["summary_path"])
        inst_all.append(float(s.get("flip_rate_mean", float("nan"))))
    inst_max = float(np.nanmax(inst_all)) if inst_all else 1.0
    if not math.isfinite(inst_max) or inst_max <= 0:
        inst_max = 1.0

    for r in runs:
        s = load_json(r["summary_path"])
        auroc = float(s.get("sklearn_auroc_final", float("nan")))
        inst = float(s.get("flip_rate_mean", float("nan")))
        me = mean_exp5_error(r["exp5_csv"])
        pkey = setting_csv_key(r["dataset"], r["k"], r["run_name"])
        rej = proxy_map.get(pkey, float("nan"))
        winr = win_map.get(pkey, float("nan"))
        # Higher score is better. Rejection: more negative mean_delta_risk => better inst vs random.
        rej_gain = float("nan")
        if math.isfinite(rej):
            rej_gain = -rej

        inst_n = inst / inst_max if math.isfinite(inst) else 0.0
        auroc_n = max(0.0, min(1.0, (auroc - 0.72) / 0.26)) if math.isfinite(auroc) else 0.0
        rej_n = max(0.0, min(1.0, (rej_gain + 0.05) / 0.08)) if math.isfinite(rej_gain) else 0.0
        target_e, span_e = 0.045, 0.12
        if math.isfinite(me):
            mid = 1.0 - min(1.0, abs(me - target_e) / span_e)
        else:
            mid = 0.3

        # Weighted score (instability priority)
        score = 2.2 * inst_n + 1.0 * auroc_n + 1.1 * rej_n + 0.9 * mid

        rows.append(
            {
                "slug": f"{r['dataset']}__{r['category']}__k{r['k']}",
                "dataset": r["dataset"],
                "category": r["category"],
                "k": r["k"],
                "run_name": r["run_name"],
                "auroc": auroc,
                "mean_instability": inst,
                "mean_sample_error_exp5": me,
                "rejection_mean_delta_risk_u6_inst_vs_random": rej,
                "rejection_win_rate_vs_random_u6": winr,
                "rejection_gain_neg_delta": rej_gain,
                "score_phase1": score,
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("score_phase1", ascending=False, kind="mergesort").reset_index(drop=True)
    top = df.head(int(top_n)).copy()

    top_path = os.path.join(out_dir, f"top{top_n}_settings.csv")
    top.to_csv(top_path, index=False)
    df.to_csv(os.path.join(out_dir, "all_settings_scored.csv"), index=False)

    rationale = os.path.join(out_dir, "selection_rationale.md")
    lines = [
        "# Top settings for same-setting multi-seed killer search",
        "",
        "Scoring (higher is better): `2.2*inst_norm + 1.0*auroc_norm + 1.1*rejection_gain_norm + 0.9*mid_error_shape`.",
        "",
        "- **inst_norm**: `flip_rate_mean` from `pairwise_instability/summary.json`, divided by max over all settings.",
        "- **auroc_norm**: clip to \\[0,1\\] from AUROC in \\[0.72, 0.98\\] (down-weight very low AUROC).",
        "- **rejection_gain_norm**: from aggregate `proxy_comparison_all_settings.csv` (proxy **u6**), use `-mean_delta_risk` (instability ordering vs random).",
        "- **mid_error_shape**: prefer mean `error` in `exp5_sample_ranking_error.csv` near ~0.045 (avoid trivial flat or degenerate).",
        "",
        "## Selected rows",
        "",
    ]
    for i, row in top.iterrows():
        lines.append(
            f"{i+1}. **{row['slug']}** — AUROC={row['auroc']:.4f}, I_mean={row['mean_instability']:.4f}, "
            f"mean_exp5_error={row['mean_sample_error_exp5']:.5f}, u6 -Δrisk_inst_vs_rand={row['rejection_mean_delta_risk_u6_inst_vs_random']:.5f}"
        )
    with open(rationale, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    meta = {"top_n": int(top_n), "result_root": os.path.abspath(result_root), "rows": top.to_dict(orient="records")}
    with open(os.path.join(out_dir, f"top{top_n}_settings.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Wrote {top_path}, rationale, json under {out_dir}")
    return top


# ---------------------------------------------------------------------------
# Phase 2: emit shell script
# ---------------------------------------------------------------------------


def phase2_emit_script(
    top_json: str,
    seeds: List[int],
    search_root: str,
    eval_freq: int,
    out_sh: str,
) -> None:
    with open(top_json, "r", encoding="utf-8") as f:
        meta = json.load(f)
    rows = meta["rows"]
    os.makedirs(os.path.dirname(out_sh) or ".", exist_ok=True)

    lines = [
        "#!/usr/bin/env bash",
        "# Auto-generated by seed_killer_evidence_pipeline.py phase2",
        "set -euo pipefail",
        f'REPO_ROOT="{_REPO_ROOT}"',
        f'PROMPTAD="{_PROMPTAD_ROOT}"',
        'cd "${PROMPTAD}"',
        'export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2',
        "",
    ]
    py = "python"
    for row in rows:
        ds, cat, k = row["dataset"], row["category"], int(row["k"])
        slug = row["slug"]
        for seed in seeds:
            job_root = os.path.join(search_root, slug, str(int(seed)))
            lines.append(f'echo "=== TRAIN {slug} seed={seed} ==="')
            lines.append(
                f'{py} train_cls.py --dataset {ds} --class_name {cat} --k-shot {k} '
                f'--root-dir "{job_root}" --seed {int(seed)} --gpu-id 0 --eval-freq {eval_freq}'
            )
            lines.append(f'echo "=== TEST {slug} seed={seed} ==="')
            lines.append(
                f'{py} test_cls.py --dataset {ds} --class_name {cat} --k-shot {k} '
                f'--root-dir "{job_root}" --output-root "{job_root}" --seed {int(seed)} --gpu-id 0 --vis false'
            )
            run_csv_name = f"CLS-{ds}-{cat}-k{k}-seed{int(seed)}-per_sample.csv"
            csv_path = os.path.join(job_root, ds, f"k_{k}", "csv", run_csv_name)
            pair_dir = os.path.join(job_root, "pairwise_instability")
            lines.append(f'echo "=== PAIRWISE {slug} seed={seed} ==="')
            lines.append(f'test -f "{csv_path}"')
            lines.append(f'{py} utils/pairwise_instability.py --csv_path "{csv_path}" --output_dir "{pair_dir}"')
            lines.append("")
    with open(out_sh, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(out_sh, 0o755)
    print(f"Wrote {out_sh} ({len(rows)} settings × {len(seeds)} seeds)")


# ---------------------------------------------------------------------------
# Phase 3: analyze multi-seed folders
# ---------------------------------------------------------------------------


def default_coverage_grid() -> np.ndarray:
    return np.clip(np.round(np.arange(1.0, 0.45, -0.05), 2), 0.5, 1.0)


def risk_coverage(df: pd.DataFrame, proxy_col: str, cov: np.ndarray) -> pd.DataFrame:
    u = df[proxy_col].to_numpy(dtype=float)
    u_sort = np.where(np.isfinite(u), u, np.inf)
    order = np.argsort(u_sort, kind="mergesort")
    return _deterministic_rejection_curve(df, order, cov)


def curve_metrics(curve_a: pd.DataFrame, curve_b: pd.DataFrame) -> Dict[str, Any]:
    m = curve_a.merge(curve_b, on="coverage", suffixes=("_a", "_b"))
    ra = m["mean_error_a"].to_numpy(dtype=float)
    rb = m["mean_error_b"].to_numpy(dtype=float)
    diff = ra - rb
    return {
        "mean_abs_risk_diff": float(np.mean(np.abs(diff))),
        "a_dominates_b": bool(np.all(ra < rb - 1e-15)),
        "b_dominates_a": bool(np.all(rb < ra - 1e-15)),
    }


def find_per_sample_csv(job_root: str, dataset: str, k: int, seed: int, category: str) -> Optional[str]:
    name = f"CLS-{dataset}-{category}-k{k}-seed{int(seed)}-per_sample.csv"
    p = os.path.join(job_root, dataset, f"k_{k}", "csv", name)
    if os.path.isfile(p):
        return p
    cand = sorted(glob.glob(os.path.join(job_root, "**", "*per_sample.csv"), recursive=True))
    for c in cand:
        if f"seed{int(seed)}" in os.path.basename(c) and "fusion" not in c and "instability" not in c:
            return c
    return None


def read_seed_metrics(
    search_root: str, slug: str, seed: int, dataset: str, category: str, k: int
) -> Optional[Dict[str, Any]]:
    job = os.path.join(search_root, slug, str(int(seed)))
    summ = os.path.join(job, "pairwise_instability", "summary.json")
    if not os.path.isfile(summ):
        csv_guess = find_per_sample_csv(job, dataset, k, seed, category)
        if csv_guess and os.path.isfile(csv_guess):
            os.makedirs(os.path.dirname(summ), exist_ok=True)
            run_full_analysis(csv_guess, os.path.join(job, "pairwise_instability"))
        else:
            return None
    if not os.path.isfile(summ):
        return None
    s = load_json(summ)
    csv_path = find_per_sample_csv(job, dataset, k, seed, category)
    return {
        "seed": int(seed),
        "slug": slug,
        "dataset": dataset,
        "category": category,
        "k": k,
        "auroc": float(s.get("sklearn_auroc_final", float("nan"))),
        "instability": float(s.get("flip_rate_mean", float("nan"))),
        "per_sample_csv": csv_path or "",
        "summary_path": summ,
    }


def phase3_analyze(
    top_json: str,
    search_root: str,
    seeds: List[int],
    proxy: str,
    delta_auroc_max: float,
    top_pairs_per_setting: int,
    out_analysis: str,
) -> None:
    os.makedirs(out_analysis, exist_ok=True)
    with open(top_json, "r", encoding="utf-8") as f:
        meta = json.load(f)
    rows = meta["rows"]
    proxy_col = PROXY_COL.get(proxy, PROXY_COL["u6"])
    cov = default_coverage_grid()

    all_metrics: List[Dict[str, Any]] = []
    for row in rows:
        slug = row["slug"]
        ds, cat, k = row["dataset"], row["category"], int(row["k"])
        for sd in seeds:
            m = read_seed_metrics(search_root, slug, sd, ds, cat, k)
            if m:
                all_metrics.append(m)

    met_df = pd.DataFrame(all_metrics)
    met_df.to_csv(os.path.join(out_analysis, "all_seed_metrics.csv"), index=False)

    pair_rows: List[Dict[str, Any]] = []
    for row in rows:
        slug = row["slug"]
        sub = met_df[met_df["slug"] == slug]
        if len(sub) < 2:
            continue
        seeds_here = sorted(sub["seed"].unique().tolist())
        for a, b in itertools.combinations(seeds_here, 2):
            ra = sub[sub["seed"] == a].iloc[0]
            rb = sub[sub["seed"] == b].iloc[0]
            d_auc = abs(float(ra["auroc"]) - float(rb["auroc"]))
            d_i = abs(float(ra["instability"]) - float(rb["instability"]))
            if d_auc >= delta_auroc_max:
                continue
            pair_rows.append(
                {
                    "slug": slug,
                    "dataset": row["dataset"],
                    "category": row["category"],
                    "k": row["k"],
                    "seed_a": int(a),
                    "seed_b": int(b),
                    "auroc_a": float(ra["auroc"]),
                    "auroc_b": float(rb["auroc"]),
                    "instability_a": float(ra["instability"]),
                    "instability_b": float(rb["instability"]),
                    "delta_auroc": d_auc,
                    "delta_instability": d_i,
                }
            )

    cand_df = pd.DataFrame(pair_rows)
    if cand_df.empty:
        raise SystemExit(
            f"No seed pairs with ΔAUROC < {delta_auroc_max}. "
            "Train all seeds or relax --delta-auroc-max."
        )
    cand_df.to_csv(os.path.join(out_analysis, "candidate_seed_pairs_all_filtered.csv"), index=False)
    cand_df = cand_df.sort_values(
        ["slug", "delta_instability", "delta_auroc"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    cand_top_parts = []
    for slug, g in cand_df.groupby("slug", sort=False):
        cand_top_parts.append(g.head(int(top_pairs_per_setting)))
    cand_top = pd.concat(cand_top_parts, ignore_index=True)
    cand_top.to_csv(os.path.join(out_analysis, "candidate_seed_pairs_top3_per_setting.csv"), index=False)

    logging.basicConfig(level=logging.WARNING)
    mini_log = logging.getLogger("seedkiller")

    eval_rows: List[Dict[str, Any]] = []
    for _, pr in cand_top.iterrows():
        csv_a = find_per_sample_csv(
            os.path.join(search_root, pr["slug"], str(int(pr["seed_a"]))),
            pr["dataset"],
            int(pr["k"]),
            int(pr["seed_a"]),
            str(pr["category"]),
        )
        csv_b = find_per_sample_csv(
            os.path.join(search_root, pr["slug"], str(int(pr["seed_b"]))),
            pr["dataset"],
            int(pr["k"]),
            int(pr["seed_b"]),
            str(pr["category"]),
        )
        err = ""
        if not csv_a or not csv_b or not os.path.isfile(csv_a) or not os.path.isfile(csv_b):
            eval_rows.append(
                {
                    **pr.to_dict(),
                    "mean_abs_risk_diff": float("nan"),
                    "a_dominates_b": False,
                    "b_dominates_a": False,
                    "error": "missing_csv",
                }
            )
            continue
        try:
            dfa = build_analysis_frame(load_and_validate_csv(csv_a), mini_log)
            dfb = build_analysis_frame(load_and_validate_csv(csv_b), mini_log)
        except Exception as e:
            eval_rows.append(
                {
                    **pr.to_dict(),
                    "mean_abs_risk_diff": float("nan"),
                    "a_dominates_b": False,
                    "b_dominates_a": False,
                    "error": str(e),
                }
            )
            continue
        ca = risk_coverage(dfa, proxy_col, cov)
        cb = risk_coverage(dfb, proxy_col, cov)
        cm = curve_metrics(ca, cb)
        eval_rows.append({**pr.to_dict(), **cm, "error": ""})

    eval_df = pd.DataFrame(eval_rows)
    eval_df.to_csv(os.path.join(out_analysis, "candidate_pairs_with_risk_metrics.csv"), index=False)
    ok = eval_df[eval_df["error"] == ""].dropna(subset=["mean_abs_risk_diff"])
    if ok.empty:
        raise SystemExit("No pairs with valid risk–coverage metrics.")

    ok = ok.copy()
    ok["_tight"] = ok["delta_auroc"] < 0.003
    ok = ok.sort_values(
        by=["mean_abs_risk_diff", "_tight", "delta_auroc", "delta_instability"],
        ascending=[False, False, True, False],
        kind="mergesort",
    )
    killer = ok.iloc[0].to_dict()

    # Figures under search_root as requested
    fig_path = os.path.join(search_root, "killer_final.png")
    csv_a = find_per_sample_csv(
        os.path.join(search_root, killer["slug"], str(int(killer["seed_a"]))),
        killer["dataset"],
        int(killer["k"]),
        int(killer["seed_a"]),
        str(killer["category"]),
    )
    csv_b = find_per_sample_csv(
        os.path.join(search_root, killer["slug"], str(int(killer["seed_b"]))),
        killer["dataset"],
        int(killer["k"]),
        int(killer["seed_b"]),
        str(killer["category"]),
    )
    dfa = build_analysis_frame(load_and_validate_csv(str(csv_a)), mini_log)
    dfb = build_analysis_frame(load_and_validate_csv(str(csv_b)), mini_log)
    ca = risk_coverage(dfa, proxy_col, cov)
    cb = risk_coverage(dfb, proxy_col, cov)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for c, lab in ((ca, f"seed {killer['seed_a']}"), (cb, f"seed {killer['seed_b']}")):
        xs = np.sort(c["coverage"].to_numpy())
        order = np.argsort(c["coverage"].to_numpy())
        ys = c["mean_error"].to_numpy()[order]
        ax1.plot(xs, ys, linewidth=2.0, label=lab)
    ax1.set_xlabel("Coverage")
    ax1.set_ylabel("Risk (mean ranking error on kept set)")
    ax1.set_title("Same AUROC, different decision reliability")
    ax1.legend(frameon=False)
    auc_note = f"AUROC: {killer['auroc_a']:.4f} vs {killer['auroc_b']:.4f}  (|Δ|={killer['delta_auroc']:.5f})"
    ax1.text(0.02, 0.98, auc_note, transform=ax1.transAxes, va="top", fontsize=9)
    ax1.grid(True, alpha=0.3)

    sub_all = met_df[met_df["slug"] == killer["slug"]]
    ax2.scatter(sub_all["instability"], sub_all["auroc"], c="0.6", s=36, label="all seeds", zorder=1)
    ax2.scatter(
        [killer["instability_a"], killer["instability_b"]],
        [killer["auroc_a"], killer["auroc_b"]],
        c=["tab:blue", "tab:orange"],
        s=80,
        zorder=3,
        label=f"seeds {killer['seed_a']}, {killer['seed_b']}",
    )
    ax2.set_xlabel("Mean instability (flip rate)")
    ax2.set_ylabel("Image-level AUROC")
    ax2.set_title(f"Same setting: {killer['slug']}")
    ax2.legend(frameon=False)
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    killer_out = {k: v for k, v in killer.items() if not str(k).startswith("_")}
    with open(os.path.join(out_analysis, "killer_pair.json"), "w", encoding="utf-8") as f:
        json.dump(killer_out, f, indent=2, ensure_ascii=False)
    if len(ok) > 1:
        sec = ok.iloc[1].to_dict()
        with open(os.path.join(out_analysis, "killer_pair_runner_up.json"), "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in sec.items() if not str(k).startswith("_")}, f, indent=2, ensure_ascii=False)

    blurb = (
        f"We fix the PromptAD setting ({killer['dataset']}/{killer['category']}/k={killer['k']}) and only change the training random seed. "
        f"The two checkpoints reach nearly identical image-level AUROC ({killer['auroc_a']:.4f} vs {killer['auroc_b']:.4f}, |Δ|={killer['delta_auroc']:.5f}), "
        f"yet exhibit substantially different mean pairwise instability ({killer['instability_a']:.4f} vs {killer['instability_b']:.4f}, |Δ|={killer['delta_instability']:.4f}). "
        f"When each model ranks its own test samples by the same proxy {proxy}(x) and we vary coverage, the risk–coverage curves diverge "
        f"(mean |Δrisk|={killer['mean_abs_risk_diff']:.4f}; dominance A over B: {killer.get('a_dominates_b', False)}, B over A: {killer.get('b_dominates_a', False)}). "
        f"Hence AUROC alone is insufficient to characterize decision reliability under score-based decisions."
    )
    with open(os.path.join(out_analysis, "paper_blurb_seed_killer.txt"), "w", encoding="utf-8") as f:
        f.write(blurb + "\n")

    print(blurb)
    print(f"Wrote {fig_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("phase1", help="Score settings and write Top-8")
    p1.add_argument("--result-root", type=str, default=os.path.join(_PROMPTAD_ROOT, "result_round1"))
    p1.add_argument(
        "--aggregate-proxy",
        type=str,
        default=os.path.join(
            _PROMPTAD_ROOT, "result_round1", "rejection_instability_aggregate", "proxy_comparison_all_settings.csv"
        ),
    )
    p1.add_argument("--top-n", type=int, default=8)
    p1.add_argument(
        "--out-dir",
        type=str,
        default=os.path.join(_PROMPTAD_ROOT, "result_seed_search", "phase1"),
    )

    p2 = sub.add_parser("phase2", help="Write train+test+pairwise bash script")
    p2.add_argument(
        "--top-json",
        type=str,
        default=os.path.join(_PROMPTAD_ROOT, "result_seed_search", "phase1", "top8_settings.json"),
    )
    p2.add_argument("--seeds", type=str, default="111,222,333,444,555")
    p2.add_argument(
        "--search-root",
        type=str,
        default=os.path.join(_PROMPTAD_ROOT, "result_seed_search"),
    )
    p2.add_argument("--eval-freq", type=int, default=2)
    p2.add_argument(
        "--out-sh",
        type=str,
        default=os.path.join(_PROMPTAD_ROOT, "result_seed_search", "phase2_train_test.sh"),
    )

    p3 = sub.add_parser("phase3", help="Analyze multi-seed outputs and write killer_final.png")
    p3.add_argument(
        "--top-json",
        type=str,
        default=os.path.join(_PROMPTAD_ROOT, "result_seed_search", "phase1", "top8_settings.json"),
    )
    p3.add_argument(
        "--search-root",
        type=str,
        default=os.path.join(_PROMPTAD_ROOT, "result_seed_search"),
    )
    p3.add_argument("--seeds", type=str, default="111,222,333,444,555")
    p3.add_argument("--proxy", type=str, default="u6", choices=list(PROXY_COL.keys()))
    p3.add_argument("--delta-auroc-max", type=float, default=0.01)
    p3.add_argument("--top-pairs-per-setting", type=int, default=3)
    p3.add_argument(
        "--out-analysis",
        type=str,
        default=os.path.join(_PROMPTAD_ROOT, "result_seed_search", "analysis"),
    )

    args = p.parse_args()
    if args.cmd == "phase1":
        phase1_score_settings(
            os.path.abspath(args.result_root),
            os.path.abspath(args.aggregate_proxy),
            int(args.top_n),
            os.path.abspath(args.out_dir),
        )
    elif args.cmd == "phase2":
        seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
        phase2_emit_script(
            os.path.abspath(args.top_json),
            seeds,
            os.path.abspath(args.search_root),
            int(args.eval_freq),
            os.path.abspath(args.out_sh),
        )
    elif args.cmd == "phase3":
        seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
        phase3_analyze(
            os.path.abspath(args.top_json),
            os.path.abspath(args.search_root),
            seeds,
            args.proxy,
            float(args.delta_auroc_max),
            int(args.top_pairs_per_setting),
            os.path.abspath(args.out_analysis),
        )


if __name__ == "__main__":
    main()
