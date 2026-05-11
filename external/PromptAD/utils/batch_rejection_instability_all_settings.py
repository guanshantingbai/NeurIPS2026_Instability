#!/usr/bin/env python3
"""
Batch-run rejection_instability_analysis.py over every per-sample CSV (each setting).

One subprocess per setting runs the full proxy pipeline (default u1..u5 in a single pass).
Optionally concatenates each setting's proxy_comparison.csv into one aggregate table.

Usage (from PromptAD repo root):

  # All settings under result_round1 (mvtec + visa, all k_*), all proxies
  python utils/batch_rejection_instability_all_settings.py \\
    --result-root result_round1

  # Subset of proxies, fixed random baseline
  python utils/batch_rejection_instability_all_settings.py \\
    --result-root result_round1 --proxies u1,u2,u3 --random-repeats 20 --seed 0

  # Custom root and aggregate path
  python utils/batch_rejection_instability_all_settings.py \\
    --result-root /path/to/result_round1 \\
    --aggregate-csv /path/to/proxy_comparison_all_settings.csv

  # Smoke test: first 3 settings only
  python utils/batch_rejection_instability_all_settings.py \\
    --result-root result_round1 --limit 3 --dry-run
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from typing import Any, List, Optional


def discover_per_sample_csvs(result_root: str) -> List[str]:
    """All *per_sample.csv files under <result_root>/**/csv/ (sorted)."""
    root = os.path.abspath(result_root)
    pattern = os.path.join(root, "**", "csv", "*per_sample.csv")
    paths = sorted(glob.glob(pattern, recursive=True))
    return [os.path.abspath(p) for p in paths if os.path.isfile(p)]


def default_output_dir_for_input(input_csv: str) -> str:
    """
    Parallel layout next to csv/:
      .../<dataset>/k_*/csv/<name>.csv
    -> .../<dataset>/k_*/rejection_instability/<stem>/
    """
    csv_dir = os.path.dirname(os.path.abspath(input_csv))
    k_dir = os.path.dirname(csv_dir)
    stem = os.path.splitext(os.path.basename(input_csv))[0]
    return os.path.join(k_dir, "rejection_instability", stem)


def run_one(
    script_path: str,
    input_csv: str,
    output_dir: str,
    proxies: str,
    random_repeats: int,
    seed: int,
    dry_run: bool,
) -> int:
    cmd = [
        sys.executable,
        script_path,
        "--input-csv",
        input_csv,
        "--output-dir",
        output_dir,
        "--proxies",
        proxies,
        "--random-repeats",
        str(random_repeats),
        "--seed",
        str(seed),
    ]
    if dry_run:
        print("[dry-run]", " ".join(cmd))
        return 0
    proc = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(script_path)))
    return int(proc.returncode)


def build_aggregate_rows(result_root: str, input_csvs: List[str]) -> Optional[Any]:
    import pandas as pd

    root = os.path.abspath(result_root)
    rows = []
    for input_csv in input_csvs:
        out_dir = default_output_dir_for_input(input_csv)
        comp = os.path.join(out_dir, "proxy_comparison.csv")
        if not os.path.isfile(comp):
            continue
        df = pd.read_csv(comp)
        rel = os.path.relpath(input_csv, root) if input_csv.startswith(root) else input_csv
        df.insert(0, "setting_csv", rel)
        rows.append(df)
    if not rows:
        return None
    return pd.concat(rows, ignore_index=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run rejection_instability_analysis on all per-sample CSV settings under a result root."
    )
    here = os.path.dirname(os.path.abspath(__file__))
    p.add_argument(
        "--analysis-script",
        type=str,
        default=os.path.join(here, "rejection_instability_analysis.py"),
        help="Path to rejection_instability_analysis.py",
    )
    p.add_argument(
        "--result-root",
        type=str,
        default="result_round1",
        help="Root folder containing **/csv/*per_sample.csv (relative to cwd if not absolute).",
    )
    p.add_argument(
        "--proxies",
        type=str,
        default="u1,u2,u3,u4,u5,u6",
        help="Forwarded to rejection_instability_analysis.py --proxies",
    )
    p.add_argument("--random-repeats", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, only process the first N settings (after sort).",
    )
    p.add_argument("--dry-run", action="store_true", help="Print commands only; no analysis, no aggregate write.")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a setting if proxy_comparison.csv already exists under its output dir.",
    )
    p.add_argument(
        "--aggregate-csv",
        type=str,
        default="",
        help="Write merged proxy_comparison over all settings. Default: <result-root>/rejection_instability_aggregate/proxy_comparison_all_settings.csv",
    )
    p.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Do not write aggregate CSV after batch.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    result_root = os.path.abspath(args.result_root)
    if not os.path.isdir(result_root):
        print(f"ERROR: result root not found: {result_root}", file=sys.stderr)
        return 1
    script_path = os.path.abspath(args.analysis_script)
    if not os.path.isfile(script_path):
        print(f"ERROR: analysis script not found: {script_path}", file=sys.stderr)
        return 1

    inputs = discover_per_sample_csvs(result_root)
    if not inputs:
        print(f"No *per_sample.csv found under {result_root}/**/csv/", file=sys.stderr)
        return 1
    if args.limit and args.limit > 0:
        inputs = inputs[: args.limit]

    failures: List[str] = []
    processed: List[str] = []

    for i, input_csv in enumerate(inputs, start=1):
        out_dir = default_output_dir_for_input(input_csv)
        if args.skip_existing:
            comp = os.path.join(out_dir, "proxy_comparison.csv")
            if os.path.isfile(comp):
                print(f"[{i}/{len(inputs)}] SKIP (exists): {input_csv}")
                processed.append(input_csv)
                continue
        print(f"[{i}/{len(inputs)}] RUN -> {out_dir}")
        rc = run_one(
            script_path,
            input_csv,
            out_dir,
            args.proxies,
            args.random_repeats,
            args.seed,
            args.dry_run,
        )
        if rc != 0:
            failures.append(input_csv)
        else:
            processed.append(input_csv)

    if failures:
        print(f"\nERROR: {len(failures)} setting(s) failed:", file=sys.stderr)
        for p in failures:
            print(f"  {p}", file=sys.stderr)

    if not args.no_aggregate and not args.dry_run:
        import pandas as pd

        agg_default = os.path.join(
            result_root, "rejection_instability_aggregate", "proxy_comparison_all_settings.csv"
        )
        agg_path = os.path.abspath(args.aggregate_csv.strip()) if args.aggregate_csv.strip() else agg_default
        os.makedirs(os.path.dirname(agg_path), exist_ok=True)
        merged = build_aggregate_rows(result_root, inputs)
        if merged is None or merged.empty:
            print("WARN: aggregate table empty (no proxy_comparison.csv found).", file=sys.stderr)
        else:
            merged.to_csv(agg_path, index=False)
            print(f"Wrote aggregate: {agg_path}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
