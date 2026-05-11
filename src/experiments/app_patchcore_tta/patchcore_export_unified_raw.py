#!/usr/bin/env python3
"""Build unified raw-score tables from PatchCore TTA per-image scores CSV."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def _sample_id(image_path: str) -> str:
    return hashlib.sha256(image_path.encode("utf-8")).hexdigest()[:24]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--dataset", type=str, default="mvtec")
    ap.add_argument("--models-run", type=str, default="", help="Recorded into config_json for traceability")
    ap.add_argument(
        "--config-extra",
        type=str,
        default="{}",
        help="JSON object merged into config_json (e.g. {\"imagesize\":224})",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.scores_csv)
    need = {"classname", "image_path", "label", "score_id", "score_flip", "score_rot_pos", "score_rot_neg"}
    if not need.issubset(df.columns):
        raise SystemExit(f"scores CSV missing columns: {sorted(need - set(df.columns))}")

    extra_cfg = json.loads(args.config_extra or "{}")
    base_cfg = {
        "pipeline": "patchcore_tta_mechanism",
        "dataset": args.dataset,
        "models_run": args.models_run,
        "source_scores_csv": str(args.scores_csv.resolve()),
    }
    base_cfg.update(extra_cfg)
    cfg_json = json.dumps(base_cfg, sort_keys=True)

    wide_rows = []
    long_rows = []
    for _, row in df.iterrows():
        sid = _sample_id(str(row["image_path"]))
        views = {
            "identity": float(row["score_id"]),
            "horizontal_flip": float(row["score_flip"]),
            "rotate_plus_5deg": float(row["score_rot_pos"]),
            "rotate_minus_5deg": float(row["score_rot_neg"]),
        }
        condition_scores = json.dumps(views)
        wide_rows.append(
            {
                "sample_id": sid,
                "label": int(row["label"]),
                "identity_score": float(row["score_id"]),
                "fused_score": float(row["score_id"]),
                "condition_scores": condition_scores,
                "view_scores": condition_scores,
                "dataset": args.dataset,
                "category": str(row["classname"]),
                "transform": "tta_bundle",
                "config": cfg_json,
                "image_path": str(row["image_path"]),
            }
        )
        for transform, score in views.items():
            long_rows.append(
                {
                    "sample_id": sid,
                    "label": int(row["label"]),
                    "dataset": args.dataset,
                    "category": str(row["classname"]),
                    "transform": transform,
                    "condition": transform,
                    "score": float(score),
                    "config": cfg_json,
                    "image_path": str(row["image_path"]),
                }
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    wide_path = args.out_dir / "unified_raw_scores.csv"
    long_path = args.out_dir / "unified_raw_scores_long.csv"
    pd.DataFrame(wide_rows).to_csv(wide_path, index=False)
    pd.DataFrame(long_rows).to_csv(long_path, index=False)
    print(f"wrote {wide_path} ({len(wide_rows)} rows)")
    print(f"wrote {long_path} ({len(long_rows)} rows)")


if __name__ == "__main__":
    main()
