#!/usr/bin/env python3
"""
Build unified PromptAD raw-score tables from existing *-per_sample.csv exports (Stage 1 mode A).

Does not run training or inference. Scans PROMPTAD-style result trees under --input-root for
``**/csv/CLS-*-per_sample.csv`` (excluding *per_sample_instability* variants).

Column resolution mirrors ``utils/pilot_instability_aware_selection.py`` / supplementary baselines
enough to support downstream Section 3.1.1 / 4 / Appendix C/G style fields (image_path, labels,
semantic / visual / fused harmonic scores). Reserved / future fields can be merged via ``config`` JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

EPS = 1e-12
FINAL_SCORE_CANDIDATES = ["harmonic_score", "final_score", "anomaly_score", "s_final", "score"]
SEM_SCORE_CANDIDATES = ["semantic_score", "s_sem", "sem_score"]
VIS_SCORE_CANDIDATES = ["visual_score", "s_vis", "vis_score"]
LABEL_CANDIDATES = ["image_label", "label", "y", "target", "gt", "is_anomaly"]
PATH_CANDIDATES = ["image_path", "path", "img_path"]


def _first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        m = lower.get(c.lower())
        if m is not None:
            return m
    return None


def _to_label01(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        x = pd.to_numeric(series, errors="coerce")
        return x.apply(lambda v: np.nan if pd.isna(v) else (1 if float(v) > 0.5 else 0))
    s = series.astype(str).str.strip().str.lower()
    pos = {"1", "anomaly", "abnormal", "positive", "pos", "true", "yes", "defect", "bad"}
    neg = {"0", "normal", "negative", "neg", "false", "no", "good"}
    return s.apply(lambda v: 1 if v in pos else (0 if v in neg else np.nan))


def _parse_cls_per_sample(path: str) -> Optional[Tuple[str, str, int, int]]:
    base = os.path.basename(path)
    m = re.match(r"CLS-([^-]+)-(.+)-k(\d+)-seed(\d+)-per_sample\.csv$", base)
    if not m:
        return None
    return m.group(1), m.group(2), int(m.group(3)), int(m.group(4))


def _sample_id(dataset: str, category: str, shot: int, seed: int, image_path: str) -> str:
    h = hashlib.sha256(f"{dataset}|{category}|{shot}|{seed}|{image_path}".encode("utf-8")).hexdigest()
    return h[:24]


def _discover_csvs(input_root: Path) -> List[Path]:
    out: List[Path] = []
    for p in input_root.rglob("*.csv"):
        if "per_sample_instability" in p.name or "fusion" in p.name:
            continue
        if not p.name.endswith("-per_sample.csv"):
            continue
        if _parse_cls_per_sample(str(p)) is None:
            continue
        out.append(p)
    return sorted(set(out))


def _load_scores(df: pd.DataFrame) -> pd.DataFrame:
    label_col = _first_existing_column(df, LABEL_CANDIDATES)
    sem_col = _first_existing_column(df, SEM_SCORE_CANDIDATES)
    vis_col = _first_existing_column(df, VIS_SCORE_CANDIDATES)
    final_col = _first_existing_column(df, FINAL_SCORE_CANDIDATES)
    img_col = _first_existing_column(df, PATH_CANDIDATES)
    if label_col is None or sem_col is None or vis_col is None:
        raise ValueError("missing label/semantic/visual columns")
    work = pd.DataFrame()
    work["image_path"] = df[img_col].astype(str) if img_col is not None else np.arange(len(df)).astype(str)
    work["image_label"] = _to_label01(df[label_col])
    work["semantic_score"] = pd.to_numeric(df[sem_col], errors="coerce")
    work["visual_score"] = pd.to_numeric(df[vis_col], errors="coerce")
    if final_col is not None:
        work["fused_score"] = pd.to_numeric(df[final_col], errors="coerce")
    else:
        work["fused_score"] = np.nan
    miss = work["fused_score"].isna()
    if miss.any():
        a = work.loc[miss, "semantic_score"].to_numpy(dtype=np.float64)
        b = work.loc[miss, "visual_score"].to_numpy(dtype=np.float64)
        work.loc[miss, "fused_score"] = (2.0 * a * b) / (a + b + EPS)
    work = work.dropna(subset=["image_label", "semantic_score", "visual_score", "fused_score"]).copy()
    work["label"] = work["image_label"].astype(int)
    work = work[(work["label"] == 0) | (work["label"] == 1)]
    return work


def _parse_filter(s: str) -> Optional[set]:
    s = (s or "").strip()
    if not s:
        return None
    return {x.strip() for x in s.split(",") if x.strip()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", type=Path, required=True, help="Root containing PromptAD result_round1/... csv trees")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--datasets-filter", type=str, default="", help="Comma subset of dataset names (mvtec,visa)")
    ap.add_argument("--classes-filter", type=str, default="", help="Comma subset of class names")
    ap.add_argument("--shots-filter", type=str, default="", help="Comma subset of k-shot integers")
    ap.add_argument("--seeds-filter", type=str, default="", help="Comma subset of seed integers")
    args = ap.parse_args()

    input_root = args.input_root.resolve()
    if not input_root.is_dir():
        raise SystemExit(f"input-root is not a directory: {input_root}")

    fd = _parse_filter(args.datasets_filter)
    fc = _parse_filter(args.classes_filter)
    fk = {int(x) for x in (_parse_filter(args.shots_filter) or set())}
    fs = {int(x) for x in (_parse_filter(args.seeds_filter) or set())}

    wide_rows: List[Dict[str, Any]] = []
    long_rows: List[Dict[str, Any]] = []
    manifest: List[Dict[str, Any]] = []

    paths = _discover_csvs(input_root)
    if not paths:
        raise SystemExit(f"no CLS-*-per_sample.csv found under {input_root}")

    for csv_path in paths:
        parsed = _parse_cls_per_sample(str(csv_path))
        if parsed is None:
            continue
        dataset, category, shot, seed = parsed
        if fd is not None and dataset not in fd:
            continue
        if fc is not None and category not in fc:
            continue
        if fk and shot not in fk:
            continue
        if fs and seed not in fs:
            continue

        df = pd.read_csv(csv_path)
        try:
            w = _load_scores(df)
        except ValueError as e:
            print(f"[skip] {csv_path}: {e}")
            continue

        cfg = {
            "pipeline": "promptad_export_unified_raw",
            "source_csv": str(csv_path.resolve()),
            "dataset": dataset,
            "category": category,
            "shot": shot,
            "seed": seed,
        }
        cfg_json = json.dumps(cfg, sort_keys=True)
        manifest.append(cfg)

        for _, row in w.iterrows():
            path = str(row["image_path"])
            sem = float(row["semantic_score"])
            vis = float(row["visual_score"])
            fus = float(row["fused_score"])
            lab = int(row["label"])
            sid = _sample_id(dataset, category, shot, seed, path)
            view_json = json.dumps({"semantic": sem, "visual": vis, "fused": fus})
            wide_rows.append(
                {
                    "sample_id": sid,
                    "label": lab,
                    "fused_score": fus,
                    "semantic_score": sem,
                    "visual_score": vis,
                    "condition": "image_level",
                    "condition_score": fus,
                    "condition_scores": view_json,
                    "dataset": dataset,
                    "category": category,
                    "shot": shot,
                    "seed": seed,
                    "config": cfg_json,
                    "image_path": path,
                }
            )
            for cond, cscore in (("semantic", sem), ("visual", vis), ("fused", fus)):
                long_rows.append(
                    {
                        "sample_id": sid,
                        "label": lab,
                        "fused_score": fus,
                        "semantic_score": sem,
                        "visual_score": vis,
                        "condition": cond,
                        "condition_score": cscore,
                        "dataset": dataset,
                        "category": category,
                        "shot": shot,
                        "seed": seed,
                        "config": cfg_json,
                        "image_path": path,
                    }
                )

    if not wide_rows:
        raise SystemExit("no rows after filters / column validation — check filters and CSV schema")

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(wide_rows).to_csv(out_dir / "unified_raw_scores_wide.csv", index=False)
    pd.DataFrame(long_rows).to_csv(out_dir / "unified_raw_scores_long.csv", index=False)
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {out_dir / 'unified_raw_scores_wide.csv'} ({len(wide_rows)} rows)")
    print(f"wrote {out_dir / 'unified_raw_scores_long.csv'} ({len(long_rows)} rows)")


if __name__ == "__main__":
    main()
