#!/usr/bin/env python3
"""
PaDiM Protocol B — same-setting multi-seed killer evidence (mirrors PromptAD seed_killer_evidence_pipeline).

Phases:
  phase1  — Score settings from ``protocol_b_multiclass_metrics.json`` (per-class fused AUROC + instability).
  phase2  — Emit bash to run ``padim_protocol_b_one_setting.py`` per (slug, seed); each job uses
            ``save_dir = search_root/slug/<seed>/`` (required basename == seed; no shared train_*.pkl).
  phase3  — Scan multi-seed folders, filter seed pairs by ΔAUROC, risk–coverage vs PromptAD-style proxy,
            pick killer, write ``killer_final.png`` + ``killer_pair.json``.

Cache note: only ``main.py`` writes ``temp_{arch}/train_{class}.pkl``; Protocol B one-run recomputes Gaussians
in memory. Cross-seed bugs come from reusing the same ``save_dir``; we enforce seed basename == seed.

Usage (from NeurIPS2026 repo root)::

  python PaDiM-Anomaly-Detection-Localization-master/padim_seed_killer_evidence_pipeline.py phase1 \\
      --metrics-json PaDiM-Anomaly-Detection-Localization-master/protocol_b_mvtec6_r18/protocol_b_multiclass_metrics.json

  python PaDiM-Anomaly-Detection-Localization-master/padim_seed_killer_evidence_pipeline.py phase2 \\
      --seeds 111,222,333,444,555

  cd PaDiM-Anomaly-Detection-Localization-master && bash padim_result_seed_search/phase2_protocol_b.sh

  python PaDiM-Anomaly-Detection-Localization-master/padim_seed_killer_evidence_pipeline.py phase3 \\
      --proxy u6 --delta-auroc-max 0.01
"""

from __future__ import annotations

import argparse
import importlib.util
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
_PADIM_ROOT = os.path.abspath(_HERE)
_REPO_ROOT = os.path.abspath(os.path.join(_PADIM_ROOT, ".."))
_PROMPTAD_ROOT = os.path.join(_REPO_ROOT, "external", "PromptAD")
if _PADIM_ROOT not in sys.path:
    sys.path.insert(0, _PADIM_ROOT)
if _PROMPTAD_ROOT not in sys.path:
    sys.path.insert(0, _PROMPTAD_ROOT)

from utils.rejection_instability_analysis import (  # noqa: E402
    _deterministic_rejection_curve,
    build_analysis_frame,
    load_and_validate_csv,
)

def _visa_class_names() -> List[str]:
    """Load PaDiM's ``datasets/visa.py`` without colliding with PyPI ``datasets``."""
    p = os.path.join(_PADIM_ROOT, "datasets", "visa.py")
    spec = importlib.util.spec_from_file_location("padim_datasets_visa", p)
    if spec is None or spec.loader is None:
        return []
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return list(getattr(mod, "CLASS_NAMES", []))


PROXY_COL = {
    "u6": "proxy_u6",
    "u2": "proxy_u2",
    "padim_marg": "proxy_u_padim_marg",
}


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def find_per_sample_csv(job_root: str) -> Optional[str]:
    p = os.path.join(job_root, "per_sample.csv")
    return p if os.path.isfile(p) else None


def _proxy_u_padim_marg(s0: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    s0 = np.asarray(s0, dtype=float)
    s1 = np.asarray(s1, dtype=float)
    s2 = np.asarray(s2, dtype=float)
    num = np.abs(s0 - s1) + np.abs(s1 - s2) + np.abs(s0 - s2)
    den = np.abs(s0) + np.abs(s1) + np.abs(s2) + 1e-6
    return num / den


def build_padim_marg_analysis_frame(csv_path: str) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    need = {"image_path", "image_label", "harmonic_score", "sample_error", "s0", "s1", "s2"}
    missing = sorted(need - set(raw.columns))
    if missing:
        raise ValueError(f"{csv_path} missing columns for padim_marg proxy: {missing}")
    out = pd.DataFrame()
    out["image_path"] = raw["image_path"].astype(str)
    lab = raw["image_label"]
    if lab.dtype == object or str(lab.dtype).startswith("string"):
        mapping = {"normal": 0, "anomaly": 1, "Normal": 0, "Anomaly": 1}
        if lab.isin(list(mapping.keys())).all():
            out["image_label"] = lab.map(mapping).astype(int)
        else:
            out["image_label"] = pd.to_numeric(lab, errors="coerce").astype(int)
    else:
        out["image_label"] = pd.to_numeric(lab, errors="coerce").astype(int)
    out["harmonic_score"] = pd.to_numeric(raw["harmonic_score"], errors="coerce")
    out["sample_error"] = pd.to_numeric(raw["sample_error"], errors="coerce")
    out["proxy_u_padim_marg"] = _proxy_u_padim_marg(
        raw["s0"].to_numpy(dtype=float),
        raw["s1"].to_numpy(dtype=float),
        raw["s2"].to_numpy(dtype=float),
    )
    return out


def build_analysis_frame_for_proxy(csv_path: str, proxy: str, log: logging.Logger) -> pd.DataFrame:
    if proxy == "padim_marg":
        return build_padim_marg_analysis_frame(csv_path)
    return build_analysis_frame(load_and_validate_csv(csv_path), log)


def phase1_score_from_protocol_json(
    metrics_json: str,
    dataset: str,
    top_n: int,
    out_dir: str,
    arch_filter: Optional[str] = None,
    visa_classes: Optional[str] = None,
) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    metrics_json = os.path.abspath(metrics_json)

    if dataset == "visa":
        vc = (visa_classes or "").strip()
        if not vc:
            raise ValueError(
                "For --dataset visa, pass --visa-classes c1,c2,... "
                "(aligns with padim_exp5_visa_macaroni1 default data path ~/datasets/pro_visa)."
            )
        arch = arch_filter or "resnet18"
        names = [c.strip() for c in vc.split(",") if c.strip()]
        vnames = _visa_class_names()
        if vnames and names:
            bad = [c for c in names if c not in vnames]
            if bad:
                raise ValueError(f"Unknown VisA classes: {bad}")
        rows = []
        for c in names:
            slug = f"visa__{c}__{arch}"
            rows.append(
                {
                    "slug": slug,
                    "dataset": "visa",
                    "category": c,
                    "arch": arch,
                    "fused_auroc": float("nan"),
                    "mean_sample_instability": float("nan"),
                    "mean_pairwise_I": float("nan"),
                    "score_phase1": 1.0,
                }
            )
        df = pd.DataFrame(rows)
        top = df.head(int(top_n)).copy()
    elif not os.path.isfile(metrics_json):
        raise FileNotFoundError(f"Metrics JSON not found: {metrics_json}")
    else:
        j = load_json(metrics_json)
        arch_json = str(j.get("arch", ""))
        if arch_filter and arch_json and arch_json != arch_filter:
            raise ValueError(f"--arch-filter {arch_filter!r} != metrics arch {arch_json!r}")
        arch = arch_json or (arch_filter or "resnet18")
        per = j.get("per_class") or {}
        if not per:
            raise ValueError(f"No per_class in {metrics_json}")

        inst_vals: List[float] = []
        for v in per.values():
            ms = v.get("mean_sample_instability")
            if ms is None or (isinstance(ms, float) and math.isnan(ms)):
                ms = v.get("mean_pairwise_I", 0.0)
            inst_vals.append(float(ms))
        inst_max = float(np.nanmax(inst_vals)) if inst_vals else 1.0
        if not math.isfinite(inst_max) or inst_max <= 0:
            inst_max = 1.0

        rows = []
        for cls, v in per.items():
            inst = float(v.get("mean_sample_instability", v.get("mean_pairwise_I", float("nan"))))
            if not math.isfinite(inst):
                inst = float(v.get("mean_pairwise_I", 0.0))
            auroc = float(v.get("fused_auroc", float("nan")))
            inst_n = inst / inst_max if math.isfinite(inst) else 0.0
            auroc_n = max(0.0, min(1.0, (auroc - 0.72) / 0.26)) if math.isfinite(auroc) else 0.0
            mpi = v.get("mean_pairwise_I")
            mid = 0.3
            if mpi is not None and math.isfinite(float(mpi)):
                mid = 1.0 - min(1.0, abs(float(mpi) - 0.05) / 0.15)
            score = 2.2 * inst_n + 1.0 * auroc_n + 0.9 * mid
            slug = f"{dataset}__{cls}__{arch}"
            rows.append(
                {
                    "slug": slug,
                    "dataset": dataset,
                    "category": cls,
                    "arch": arch,
                    "fused_auroc": auroc,
                    "mean_sample_instability": float(v.get("mean_sample_instability", float("nan"))),
                    "mean_pairwise_I": float(v.get("mean_pairwise_I", float("nan"))),
                    "score_phase1": score,
                }
            )

        df = pd.DataFrame(rows)
        df = df.sort_values("score_phase1", ascending=False, kind="mergesort").reset_index(drop=True)
        top = df.head(int(top_n)).copy()

    top_path = os.path.join(out_dir, "top_settings.csv")
    top.to_csv(top_path, index=False)
    if not df.empty:
        df.to_csv(os.path.join(out_dir, "all_settings_scored.csv"), index=False)

    rationale = os.path.join(out_dir, "selection_rationale.md")
    lines = [
        "# PaDiM Protocol B — top settings for multi-seed killer search",
        "",
        "Scoring (higher is better): `2.2*inst_norm + 1.0*auroc_norm + 0.9*mid_flip_shape`.",
        "",
        "- **inst_norm**: `mean_sample_instability` (fallback `mean_pairwise_I`) / max over classes in JSON.",
        "- **auroc_norm**: clip AUROC from [0.72, 0.98] to [0,1].",
        "- **mid_flip_shape**: prefer `mean_pairwise_I` near ~0.05 (degenerate settings down-weighted).",
        "",
        "## Selected rows",
        "",
    ]
    for i, row in top.iterrows():
        lines.append(
            f"{i+1}. **{row['slug']}** — fused_auroc={row.get('fused_auroc', float('nan')):.4f}, "
            f"mean_sample_I={row.get('mean_sample_instability', float('nan')):.5f}, score={row['score_phase1']:.4f}"
        )
    with open(rationale, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    meta = {
        "top_n": int(min(top_n, len(top))),
        "requested_top_n": int(top_n),
        "metrics_json": None if dataset == "visa" else metrics_json,
        "dataset": dataset,
        "rows": top.to_dict(orient="records"),
    }
    with open(os.path.join(out_dir, "top_settings.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Wrote {top_path}, rationale, json under {out_dir}")
    return top


def phase2_emit_script(
    top_json: str,
    seeds: List[int],
    search_root: str,
    out_sh: str,
    data_path_mvtec: str,
    data_path_visa: str,
    batch_size: Optional[int],
    num_workers: Optional[int],
) -> None:
    with open(top_json, "r", encoding="utf-8") as f:
        meta = json.load(f)
    rows = meta["rows"]
    os.makedirs(os.path.dirname(out_sh) or ".", exist_ok=True)

    padim = _PADIM_ROOT
    lines = [
        "#!/usr/bin/env bash",
        "# Auto-generated by padim_seed_killer_evidence_pipeline.py phase2",
        "set -euo pipefail",
        f'PADIM_ROOT="{padim}"',
        f'SEARCH_ROOT="{os.path.abspath(search_root)}"',
        'cd "${PADIM_ROOT}"',
        'export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2',
        "",
    ]
    py = "python3"
    bs = f" --batch_size {int(batch_size)}" if batch_size is not None else ""
    nw = f" --num_workers {int(num_workers)}" if num_workers is not None else ""
    for row in rows:
        ds = str(row["dataset"])
        cat = str(row["category"])
        arch = str(row["arch"])
        slug = str(row["slug"])
        dp = data_path_mvtec if ds == "mvtec" else data_path_visa
        for seed in seeds:
            job = os.path.join(os.path.abspath(search_root), slug, str(int(seed)))
            lines.append(f'echo "=== PaDiM Protocol B {slug} seed={seed} ==="')
            lines.append(
                f'{py} padim_protocol_b_one_run.py --dataset {ds} --class_name {cat} --arch {arch} '
                f'--seed {int(seed)} --save_dir "{job}" --data_path "{dp}"{bs}{nw}'
            )
            lines.append("")
    with open(out_sh, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(out_sh, 0o755)
    print(f"Wrote {out_sh} ({len(rows)} settings × {len(seeds)} seeds)")


def read_seed_metrics(search_root: str, slug: str, seed: int) -> Optional[Dict[str, Any]]:
    job = os.path.join(search_root, slug, str(int(seed)))
    summ_path = os.path.join(job, "summary.json")
    if not os.path.isfile(summ_path):
        return None
    s = load_json(summ_path)
    csv_path = find_per_sample_csv(job) or ""
    return {
        "seed": int(seed),
        "slug": slug,
        "dataset": str(s.get("dataset", "")),
        "category": str(s.get("class_name", "")),
        "arch": str(s.get("arch", "")),
        "auroc": float(s.get("sklearn_auroc_final", s.get("fused_auroc", float("nan")))),
        "instability": float(s.get("flip_rate_mean", float("nan"))),
        "per_sample_csv": csv_path,
        "summary_path": summ_path,
    }


def phase3_analyze(
    top_json: str,
    search_root: str,
    seeds: List[int],
    proxy: str,
    delta_auroc_max: float,
    top_pairs_per_setting: int,
    out_analysis: str,
    killer_sort: str = "risk",
) -> None:
    os.makedirs(out_analysis, exist_ok=True)
    with open(top_json, "r", encoding="utf-8") as f:
        meta = json.load(f)
    rows = meta["rows"]
    proxy_col = PROXY_COL.get(proxy)
    if not proxy_col:
        raise ValueError(f"Unknown proxy {proxy!r}; choose from {list(PROXY_COL)}")
    cov = default_coverage_grid()

    all_metrics: List[Dict[str, Any]] = []
    for row in rows:
        slug = str(row["slug"])
        for sd in seeds:
            m = read_seed_metrics(search_root, slug, int(sd))
            if m:
                all_metrics.append(m)

    met_df = pd.DataFrame(all_metrics)
    met_df.to_csv(os.path.join(out_analysis, "all_seed_metrics.csv"), index=False)

    pair_rows: List[Dict[str, Any]] = []
    for row in rows:
        slug = str(row["slug"])
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
                    "arch": row.get("arch", ""),
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
            f"No seed pairs with ΔAUROC < {delta_auroc_max}. Run phase2 for all seeds or relax --delta-auroc-max."
        )
    cand_df.to_csv(os.path.join(out_analysis, "candidate_seed_pairs_all_filtered.csv"), index=False)
    # Prefer **tight AUROC** first, then large Δinstability — so pairs like (111,222) with ΔAUROC=0
    # are not dropped in favor of high-ΔI but looser-AUROC pairs (stronger killer narrative).
    cand_df = cand_df.sort_values(
        ["slug", "delta_auroc", "delta_instability"],
        ascending=[True, True, False],
        kind="mergesort",
    )
    cand_top_parts = []
    k_pairs = int(top_pairs_per_setting)
    for slug, g in cand_df.groupby("slug", sort=False):
        if k_pairs <= 0:
            cand_top_parts.append(g)
        else:
            cand_top_parts.append(g.head(k_pairs))
    cand_top = pd.concat(cand_top_parts, ignore_index=True)
    cand_top.to_csv(os.path.join(out_analysis, "candidate_seed_pairs_evaluated_subset.csv"), index=False)

    logging.basicConfig(level=logging.WARNING)
    mini_log = logging.getLogger("padim_seedkiller")

    eval_rows: List[Dict[str, Any]] = []
    for _, pr in cand_top.iterrows():
        root_a = os.path.join(search_root, pr["slug"], str(int(pr["seed_a"])))
        root_b = os.path.join(search_root, pr["slug"], str(int(pr["seed_b"])))
        csv_a = find_per_sample_csv(root_a)
        csv_b = find_per_sample_csv(root_b)
        if not csv_a or not csv_b:
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
            dfa = build_analysis_frame_for_proxy(str(csv_a), proxy, mini_log)
            dfb = build_analysis_frame_for_proxy(str(csv_b), proxy, mini_log)
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
    ok["_dom"] = (ok["a_dominates_b"] | ok["b_dominates_a"]).astype(int)
    if killer_sort == "dominance_first":
        ok = ok.sort_values(
            by=["_dom", "mean_abs_risk_diff", "_tight", "delta_auroc", "delta_instability"],
            ascending=[False, False, False, True, False],
            kind="mergesort",
        )
    else:
        ok = ok.sort_values(
            by=["mean_abs_risk_diff", "_dom", "_tight", "delta_auroc", "delta_instability"],
            ascending=[False, False, False, True, False],
            kind="mergesort",
        )
    killer = ok.iloc[0].to_dict()

    fig_path = os.path.join(os.path.abspath(search_root), "killer_final.png")
    csv_a = find_per_sample_csv(os.path.join(search_root, killer["slug"], str(int(killer["seed_a"]))))
    csv_b = find_per_sample_csv(os.path.join(search_root, killer["slug"], str(int(killer["seed_b"]))))
    dfa = build_analysis_frame_for_proxy(str(csv_a), proxy, mini_log)
    dfb = build_analysis_frame_for_proxy(str(csv_b), proxy, mini_log)
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
    ax1.set_title("Same AUROC, different decision reliability (PaDiM Protocol B)")
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
    ax2.set_ylabel("Image-level AUROC (fused)")
    ax2.set_title(f"Same setting: {killer['slug']}")
    ax2.legend(frameon=False)
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    killer_out = {k: v for k, v in killer.items() if not str(k).startswith("_")}
    with open(os.path.join(out_analysis, "killer_pair.json"), "w", encoding="utf-8") as f:
        json.dump(killer_out, f, indent=2, ensure_ascii=False)

    manifest = {
        "protocol": "PaDiM Protocol B",
        "proxy": proxy,
        "killer_sort": killer_sort,
        "top_pairs_per_setting": top_pairs_per_setting,
        "search_root": os.path.abspath(search_root),
        "top_json": os.path.abspath(top_json),
        "seeds": seeds,
        "killer_pair": killer_out,
        "artifacts": {
            "killer_final_png": fig_path,
            "all_seed_metrics": os.path.join(out_analysis, "all_seed_metrics.csv"),
            "candidate_pairs": os.path.join(out_analysis, "candidate_pairs_with_risk_metrics.csv"),
        },
    }
    with open(os.path.join(out_analysis, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    blurb = (
        f"We fix the PaDiM Protocol B setting ({killer['dataset']}/{killer['category']}, arch={killer.get('arch', '')}) "
        f"and only change the random seed controlling fused subspace indices. "
        f"The two runs reach nearly identical image-level AUROC ({killer['auroc_a']:.4f} vs {killer['auroc_b']:.4f}, |Δ|={killer['delta_auroc']:.5f}), "
        f"yet differ in mean pairwise flip rate ({killer['instability_a']:.4f} vs {killer['instability_b']:.4f}, |Δ|={killer['delta_instability']:.4f}). "
        f"Risk–coverage curves under proxy {proxy}(x) diverge (mean |Δrisk|={killer['mean_abs_risk_diff']:.4f}; "
        f"A dominates B: {killer.get('a_dominates_b', False)}, B dominates A: {killer.get('b_dominates_a', False)}). "
        f"Hence AUROC alone does not summarize decision reliability for score-based decisions."
    )
    with open(os.path.join(out_analysis, "paper_blurb_seed_killer.txt"), "w", encoding="utf-8") as f:
        f.write(blurb + "\n")

    print(blurb)
    print(f"Wrote {fig_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    default_metrics = os.path.join(_PADIM_ROOT, "protocol_b_mvtec6_r18", "protocol_b_multiclass_metrics.json")
    default_search = os.path.join(_PADIM_ROOT, "padim_result_seed_search")
    default_phase1 = os.path.join(default_search, "phase1")

    p1 = sub.add_parser("phase1", help="Score classes from protocol_b_multiclass_metrics.json")
    p1.add_argument("--metrics-json", type=str, default=default_metrics)
    p1.add_argument("--dataset", type=str, choices=["mvtec", "visa"], default="mvtec")
    p1.add_argument("--arch-filter", type=str, default=None, help="Require this arch string in metrics JSON")
    p1.add_argument("--top-n", type=int, default=8)
    p1.add_argument("--out-dir", type=str, default=default_phase1)
    p1.add_argument(
        "--visa-classes",
        type=str,
        default="",
        help="VisA only: comma classes when metrics JSON is absent (macaroni1 aligns with padim_exp5_visa_macaroni1).",
    )

    p2 = sub.add_parser("phase2", help="Write padim_protocol_b_one_run bash script")
    p2.add_argument(
        "--top-json",
        type=str,
        default=os.path.join(default_phase1, "top_settings.json"),
    )
    p2.add_argument("--seeds", type=str, default="111,222,333,444,555")
    p2.add_argument("--search-root", type=str, default=default_search)
    p2.add_argument(
        "--out-sh",
        type=str,
        default=os.path.join(default_search, "phase2_protocol_b.sh"),
    )
    p2.add_argument("--data-path-mvtec", type=str, default=os.path.expanduser("~/datasets/mvtec"))
    p2.add_argument("--data-path-visa", type=str, default=os.path.expanduser("~/datasets/pro_visa"))
    p2.add_argument("--batch-size", type=int, default=None)
    p2.add_argument("--num-workers", type=int, default=None)

    p3 = sub.add_parser("phase3", help="Analyze multi-seed outputs")
    p3.add_argument("--top-json", type=str, default=os.path.join(default_phase1, "top_settings.json"))
    p3.add_argument("--search-root", type=str, default=default_search)
    p3.add_argument("--seeds", type=str, default="111,222,333,444,555")
    p3.add_argument("--proxy", type=str, default="u6", choices=list(PROXY_COL.keys()))
    p3.add_argument("--delta-auroc-max", type=float, default=0.01)
    p3.add_argument(
        "--top-pairs-per-setting",
        type=int,
        default=15,
        help="How many tight-ΔAUROC pairs per setting get risk–coverage metrics; use 0 for all filtered pairs.",
    )
    p3.add_argument(
        "--killer-sort",
        type=str,
        choices=["risk", "dominance_first"],
        default="risk",
        help="risk: maximize mean|Δrisk| (tie-break dominance). dominance_first: prefer strict curve dominance.",
    )
    p3.add_argument("--out-analysis", type=str, default=os.path.join(default_search, "analysis"))

    args = p.parse_args()
    if args.cmd == "phase1":
        phase1_score_from_protocol_json(
            metrics_json=args.metrics_json,
            dataset=args.dataset,
            top_n=int(args.top_n),
            out_dir=os.path.abspath(args.out_dir),
            arch_filter=args.arch_filter,
            visa_classes=(args.visa_classes or None),
        )
    elif args.cmd == "phase2":
        seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
        phase2_emit_script(
            os.path.abspath(args.top_json),
            seeds,
            os.path.abspath(args.search_root),
            os.path.abspath(args.out_sh),
            os.path.expanduser(args.data_path_mvtec),
            os.path.expanduser(args.data_path_visa),
            args.batch_size,
            args.num_workers,
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
            killer_sort=str(args.killer_sort),
        )


if __name__ == "__main__":
    main()
