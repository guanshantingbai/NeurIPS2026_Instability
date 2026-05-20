#!/usr/bin/env python3
"""
Stage 2 only: aggregate PromptAD pairwise metrics from unified raw scores.

Reuses the same vectorized engine as Section 3/4 analyze scripts; optionally
saves the full pairwise table for strict pairwise-level conditioning in Section 4
mechanism figures.

Inputs (cached, no Stage 1 rerun):
  outputs/cached_results/raw_scores/promptad/unified_raw_scores_wide.csv
  (or unified_raw_scores_long.csv as fallback)

Outputs:
  <out-dir>/setting_level_metrics.csv
  <out-dir>/pairwise_metrics.csv          (if --save-pairwise)
  <out-dir>/controlled_margin_detail.csv
  <out-dir>/controlled_margin_analysis.csv
  <out-dir>/aggregation_done.json

Plus mirrors under outputs/cached_results/sec3_promptad and sec4_systematic so
existing downstream scripts can find pairwise_metrics.csv and margin tables
without further wiring.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.core.pairwise_aggregation import (
    aggregate_pairwise_from_raw,
    controlled_margin_rows,
    load_wide_or_long,
    write_aggregation_done,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("outputs/cached_results/raw_scores/promptad"),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/cached_results/promptad_pairwise"),
    )
    ap.add_argument(
        "--mirror-dirs",
        type=Path,
        nargs="*",
        default=[
            Path("outputs/cached_results/sec3_promptad"),
            Path("outputs/cached_results/sec4_systematic"),
        ],
        help="Copy pairwise_metrics.csv + margin tables to these dirs for downstream consumers.",
    )
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-pairs-per-setting", type=int, default=None)
    ap.add_argument("--pair-sampling-seed", type=int, default=42)
    ap.add_argument("--save-pairwise", action="store_true")
    ap.add_argument(
        "--include-image-paths",
        action="store_true",
        help="Include anomaly/normal image_path and sample_id columns in pairwise output (heavier).",
    )
    args = ap.parse_args()

    raw_dir = args.raw_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_p = raw_dir / "unified_raw_scores_wide.csv"
    if not raw_p.is_file():
        raw_p = raw_dir / "unified_raw_scores_long.csv"

    print("[pairwise_aggregation] Plan")
    print(f"  raw input         : {raw_p}")
    print(f"  out dir           : {out_dir}")
    print(f"  mirror dirs       : {[str(p) for p in args.mirror_dirs]}")
    print(f"  workers           : {args.workers}")
    print(f"  save_pairwise     : {args.save_pairwise}")
    print(f"  include_paths     : {args.include_image_paths}")
    if args.save_pairwise:
        print("  WARNING: pairwise_metrics.csv can be large (~15M rows on full PromptAD universe).")

    t0 = time.time()
    df = load_wide_or_long(raw_dir)
    print(f"[pairwise_aggregation] Loaded {len(df)} sample rows ({time.time() - t0:.1f}s)")

    n_settings_expected = df.groupby(["dataset", "category", "shot", "seed"], sort=False).ngroups
    print(f"[pairwise_aggregation] expected settings: {n_settings_expected}")

    t1 = time.time()
    setting_df, pairwise_df, meta = aggregate_pairwise_from_raw(
        df,
        workers=int(args.workers),
        max_pairs_per_setting=args.max_pairs_per_setting,
        pair_sampling_seed=int(args.pair_sampling_seed),
        save_pairwise=bool(args.save_pairwise),
        include_image_paths=bool(args.include_image_paths),
    )
    print(
        f"[pairwise_aggregation] aggregation done in {time.time() - t1:.1f}s: "
        f"settings={meta['n_settings']}, pairs_written={meta['n_pairs_written']}, "
        f"pairs_raw={meta['total_pairs_enumerated_raw']}"
    )

    setting_path = out_dir / "setting_level_metrics.csv"
    setting_df.to_csv(setting_path, index=False)
    pairwise_path = out_dir / "pairwise_metrics.csv"
    if args.save_pairwise and len(pairwise_df):
        pairwise_df.to_csv(pairwise_path, index=False)

    long_cm = pd.DataFrame()
    cm = pd.DataFrame()
    if len(pairwise_df):
        long_cm, cm = controlled_margin_rows(pairwise_df)
        (out_dir / "controlled_margin_detail.csv").write_text(
            long_cm.to_csv(index=False), encoding="utf-8"
        )
        (out_dir / "controlled_margin_analysis.csv").write_text(
            cm.to_csv(index=False), encoding="utf-8"
        )

    write_aggregation_done(
        out_dir,
        raw_dir=raw_dir,
        meta={
            "generation_time": datetime.now(timezone.utc).isoformat(),
            "controlled_margin_detail_rows": int(len(long_cm)),
            "controlled_margin_aggregated_rows": int(len(cm)),
            **meta,
        },
        pairwise_path=pairwise_path if args.save_pairwise and len(pairwise_df) else None,
    )

    for mirror in args.mirror_dirs:
        m = mirror.resolve()
        m.mkdir(parents=True, exist_ok=True)
        setting_df.to_csv(m / "setting_level_metrics.csv", index=False)
        if args.save_pairwise and len(pairwise_df):
            pairwise_df.to_csv(m / "pairwise_metrics.csv", index=False)
        if len(long_cm):
            long_cm.to_csv(m / "controlled_margin_detail.csv", index=False)
        if len(cm):
            cm.to_csv(m / "controlled_margin_analysis.csv", index=False)

    print(f"[pairwise_aggregation] wrote {setting_path}")
    if args.save_pairwise:
        print(f"[pairwise_aggregation] wrote {pairwise_path}")
    print("[pairwise_aggregation] mirrored to:", [str(p) for p in args.mirror_dirs])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
