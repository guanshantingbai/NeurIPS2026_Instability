#!/usr/bin/env python3
"""Aggregate Protocol B ``per_sample.csv`` + ``summary.json`` into unified raw scores and downstream CSVs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _sample_id(image_path: str, dataset: str, category: str, backbone: str, seed: int) -> str:
    h = hashlib.sha256(
        f"{dataset}|{category}|{backbone}|{seed}|{image_path}".encode("utf-8")
    ).hexdigest()
    return h[:24]


def _parse_slug(slug: str) -> Tuple[str, str, str]:
    parts = slug.split("__")
    if len(parts) != 3:
        raise ValueError(f"expected slug form dataset__class__arch, got {slug!r}")
    return parts[0], parts[1], parts[2]


def discover_jobs(jobs_root: Path) -> List[Path]:
    out: List[Path] = []
    for seed_dir in sorted(jobs_root.glob("*/*")):
        if not seed_dir.is_dir():
            continue
        if not seed_dir.name.isdigit():
            continue
        csv_p = seed_dir / "per_sample.csv"
        if csv_p.is_file():
            out.append(seed_dir)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs-root", type=Path, required=True, help="Root containing <dataset>__<class>__<arch>/<seed>/")
    ap.add_argument(
        "--raw-out",
        type=Path,
        default=Path("outputs/cached_results/raw_scores/padim"),
        help="Writes unified_raw_scores_long.csv / unified_raw_scores_wide.csv + manifest.json",
    )
    ap.add_argument(
        "--sec3-out",
        type=Path,
        default=Path("outputs/cached_results/sec3_padim"),
        help="Writes marginal_protocol_b.csv for Section 3.1.2",
    )
    ap.add_argument(
        "--appendix-e-out",
        type=Path,
        default=Path("outputs/cached_results/app_padim_representation"),
        help="Writes mechanism_from_raw.csv for Appendix E tables",
    )
    args = ap.parse_args()

    jobs_root = args.jobs_root.resolve()
    if not jobs_root.is_dir():
        raise SystemExit(f"jobs-root is not a directory: {jobs_root}")

    raw_out = args.raw_out.resolve()
    sec3_out = args.sec3_out.resolve()
    app_out = args.appendix_e_out.resolve()
    raw_out.mkdir(parents=True, exist_ok=True)
    sec3_out.mkdir(parents=True, exist_ok=True)
    app_out.mkdir(parents=True, exist_ok=True)

    job_dirs = discover_jobs(jobs_root)
    if not job_dirs:
        raise SystemExit(f"no Protocol B job dirs with per_sample.csv under {jobs_root}")

    long_rows: List[Dict[str, Any]] = []
    wide_rows: List[Dict[str, Any]] = []
    marginal_rows: List[Dict[str, Any]] = []
    summaries_by_setting: Dict[str, List[Dict[str, Any]]] = {}
    manifest: List[Dict[str, Any]] = []

    views = ("s0", "s1", "s2", "s_fused")

    for seed_dir in job_dirs:
        slug = seed_dir.parent.name
        dataset, category, backbone = _parse_slug(slug)
        seed = int(seed_dir.name)
        per_csv = seed_dir / "per_sample.csv"
        sum_json = seed_dir / "summary.json"
        if not sum_json.is_file():
            raise SystemExit(f"missing summary.json next to per_sample: {seed_dir}")

        with open(sum_json, encoding="utf-8") as f:
            summary = json.load(f)

        setting = f"{dataset}__{category}__{backbone}"
        marginal_rows.append(
            {
                "setting": setting,
                "seed": seed,
                "auroc": float(summary.get("fused_auroc", summary.get("sklearn_auroc_final", float("nan")))),
                "instability": float(summary.get("mean_sample_instability", float("nan"))),
            }
        )
        summaries_by_setting.setdefault(setting, []).append(
            {"seed": seed, "auroc": float(summary.get("fused_auroc", summary.get("sklearn_auroc_final", float("nan"))))}
        )

        cfg = {
            "pipeline": "padim_protocol_b_one_run",
            "dataset": dataset,
            "category": category,
            "backbone": backbone,
            "seed": seed,
            "job_dir": str(seed_dir),
            "flip_rate_mean": summary.get("flip_rate_mean"),
            "n_test": summary.get("n_test"),
        }
        cfg_json = json.dumps(cfg, sort_keys=True)
        manifest.append(cfg)

        df = pd.read_csv(per_csv)
        need = {"image_path", "image_label", "s0", "s1", "s2", "s_fused"}
        if not need.issubset(df.columns):
            raise SystemExit(f"{per_csv} missing columns: {sorted(need - set(df.columns))}")

        for _, row in df.iterrows():
            lab_s = str(row["image_label"]).strip().lower()
            label = 1 if lab_s == "anomaly" else 0
            path = str(row["image_path"])
            sid = _sample_id(path, dataset, category, backbone, seed)
            fused = float(row["s_fused"])
            view_map = {v: float(row[v]) for v in views}
            view_json = json.dumps(view_map)
            wide_rows.append(
                {
                    "sample_id": sid,
                    "label": label,
                    "fused_score": fused,
                    "view_scores": view_json,
                    "dataset": dataset,
                    "category": category,
                    "backbone": backbone,
                    "seed": seed,
                    "config": cfg_json,
                    "image_path": path,
                }
            )
            for vid in views:
                long_rows.append(
                    {
                        "sample_id": sid,
                        "label": label,
                        "fused_score": fused,
                        "view_id": vid,
                        "condition": vid,
                        "condition_score": float(row[vid]),
                        "dataset": dataset,
                        "category": category,
                        "backbone": backbone,
                        "seed": seed,
                        "config": cfg_json,
                        "image_path": path,
                    }
                )

    pd.DataFrame(long_rows).to_csv(raw_out / "unified_raw_scores_long.csv", index=False)
    pd.DataFrame(wide_rows).to_csv(raw_out / "unified_raw_scores_wide.csv", index=False)
    with open(raw_out / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    mdf = pd.DataFrame(marginal_rows)
    mdf.to_csv(sec3_out / "marginal_protocol_b.csv", index=False)

    per_setting_mean: List[float] = []
    per_setting_max: List[float] = []
    for _setting, lst in summaries_by_setting.items():
        seeds = sorted(lst, key=lambda x: x["seed"])
        aucs = [x["auroc"] for x in seeds]
        if len(aucs) >= 2:
            deltas = [abs(aucs[i] - aucs[j]) for i in range(len(aucs)) for j in range(i + 1, len(aucs))]
            per_setting_mean.append(float(np.mean(deltas)))
            per_setting_max.append(float(np.max(deltas)))

    g_mean = float(np.nanmean(per_setting_mean)) if per_setting_mean else float("nan")
    g_max = float(np.max(per_setting_max)) if per_setting_max else float("nan")
    mech_rows = [
        {"step": "cross_seed_mean_abs_delta_auroc", "mean_delta_auroc": g_mean},
        {"step": "cross_seed_max_pairwise_delta_auroc", "mean_delta_auroc": g_max},
    ]
    mech_df = pd.DataFrame(mech_rows)
    mech_path = app_out / "mechanism_from_raw.csv"
    mech_df.to_csv(mech_path, index=False)

    tables_dir = _REPO_ROOT / "outputs" / "tables" / "app_padim_representation"
    tables_dir.mkdir(parents=True, exist_ok=True)
    mech_df.to_csv(tables_dir / "mechanism_from_raw.csv", index=False)

    print(f"wrote {raw_out / 'unified_raw_scores_long.csv'} ({len(long_rows)} rows)")
    print(f"wrote {raw_out / 'unified_raw_scores_wide.csv'} ({len(wide_rows)} rows)")
    print(f"wrote {sec3_out / 'marginal_protocol_b.csv'} ({len(mdf)} rows)")
    print(f"wrote {mech_path}")


if __name__ == "__main__":
    main()
