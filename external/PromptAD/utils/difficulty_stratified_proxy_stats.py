#!/usr/bin/env python3
"""
Stratify u2 / u6 proxy_comparison metrics by per-setting difficulty (mean sample_error).

Reads proxy_comparison_all_settings.csv + per_sample_instability_analysis.csv per setting.
Prints a summary table and writes difficulty_analysis.csv next to the input aggregate.
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths (edit input_csv if needed)
# ---------------------------------------------------------------------------

_PROMPTAD_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
input_csv = os.path.join(
    _PROMPTAD_ROOT,
    "result_round1",
    "rejection_instability_aggregate",
    "proxy_comparison_all_settings.csv",
)
RESULT_ROOT = os.path.normpath(os.path.join(os.path.dirname(input_csv), ".."))
output_csv = os.path.join(os.path.dirname(input_csv), "difficulty_analysis.csv")

PROXIES = ("u2", "u6")
BIN_ORDER = ["easy", "medium", "hard"]


def difficulty_bin(e: float) -> str:
    if e < 0.05:
        return "easy"
    if e < 0.15:
        return "medium"
    return "hard"


def instability_analysis_csv(result_root: str, setting_csv: str) -> str:
    stem = os.path.splitext(os.path.basename(setting_csv))[0]
    full_input = os.path.join(result_root, setting_csv)
    k_dir = os.path.dirname(os.path.dirname(full_input))
    return os.path.join(k_dir, "rejection_instability", stem, "per_sample_instability_analysis.csv")


def load_mean_sample_error(path: str) -> float:
    d = pd.read_csv(path)
    if "sample_error" not in d.columns:
        raise ValueError("no column sample_error")
    return float(d["sample_error"].mean())


def main() -> int:
    if not os.path.isfile(input_csv):
        print(f"ERROR: not found: {input_csv}", file=sys.stderr)
        return 1

    df = pd.read_csv(input_csv)
    need = [
        "setting_csv",
        "proxy_name",
        "mean_delta_risk",
        "mean_delta_risk_inst_vs_score",
    ]
    miss = [c for c in need if c not in df.columns]
    if miss:
        print(f"ERROR: missing columns {miss}", file=sys.stderr)
        return 1

    df = df[df["proxy_name"].isin(PROXIES)].copy()
    if df.empty:
        print("ERROR: no u2/u6 rows.", file=sys.stderr)
        return 1

    # One mean_sample_error per setting (same for both proxies)
    setting_to_e: dict[str, float] = {}
    for s in df["setting_csv"].unique():
        apath = instability_analysis_csv(RESULT_ROOT, str(s))
        try:
            setting_to_e[str(s)] = load_mean_sample_error(apath)
        except FileNotFoundError:
            warnings.warn(f"missing file, skip setting: {apath}")
            continue
        except Exception as ex:
            warnings.warn(f"read failed ({ex}), skip: {apath}")
            continue

    df["mean_sample_error"] = df["setting_csv"].map(setting_to_e)
    df = df.dropna(subset=["mean_sample_error"])
    if df.empty:
        print("ERROR: no rows after loading per-setting errors.", file=sys.stderr)
        return 1

    df["difficulty_bin"] = df["mean_sample_error"].map(difficulty_bin)
    df["difficulty_bin"] = pd.Categorical(df["difficulty_bin"], categories=BIN_ORDER, ordered=True)

    def _win_neg(s: pd.Series) -> float:
        return float((s.astype(float) < 0).mean())

    gcols = ["proxy_name", "difficulty_bin"]
    out = (
        df.groupby(gcols, observed=True)
        .agg(
            n=("mean_delta_risk", "count"),
            mean_delta_risk_mean=("mean_delta_risk", lambda s: float(np.mean(s.astype(float)))),
            mean_delta_risk_std=("mean_delta_risk", lambda s: float(np.std(s.astype(float), ddof=0))),
            win_rate=("mean_delta_risk", _win_neg),
            mean_delta_risk_inst_vs_score_mean=(
                "mean_delta_risk_inst_vs_score",
                lambda s: float(np.nanmean(s.astype(float))),
            ),
            win_rate_vs_score=("mean_delta_risk_inst_vs_score", _win_neg),
        )
        .reset_index()
    )
    out["difficulty_bin"] = pd.Categorical(out["difficulty_bin"], categories=BIN_ORDER, ordered=True)
    out = out.sort_values(["proxy_name", "difficulty_bin"]).reset_index(drop=True)

    # Pretty print
    disp = out.copy()
    disp["n"] = disp["n"].astype(int)
    disp["mean_delta_risk_mean"] = disp["mean_delta_risk_mean"].map(lambda x: f"{x:.6f}")
    disp["mean_delta_risk_std"] = disp["mean_delta_risk_std"].map(lambda x: f"{x:.6f}")
    disp["win_rate"] = disp["win_rate"].map(lambda x: f"{x:.4f}")
    disp["mean_delta_risk_inst_vs_score_mean"] = disp["mean_delta_risk_inst_vs_score_mean"].map(lambda x: f"{x:.6f}")
    disp["win_rate_vs_score"] = disp["win_rate_vs_score"].map(lambda x: f"{x:.4f}")

    print("\n=== Difficulty-stratified summary (inst vs random / inst vs score) ===\n")
    print(
        disp.rename(
            columns={
                "proxy_name": "proxy",
                "difficulty_bin": "difficulty",
                "n": "n",
                "mean_delta_risk_mean": "mean_d_risk",
                "mean_delta_risk_std": "std_d_risk",
                "win_rate": "win_rate",
                "mean_delta_risk_inst_vs_score_mean": "mean_d_risk_vs_score",
                "win_rate_vs_score": "win_rate_vs_score",
            }
        ).to_string(index=False)
    )
    print()

    # Save numeric (full precision)
    out_save = out.rename(
        columns={
            "mean_delta_risk_mean": "mean_delta_risk_mean",
            "mean_delta_risk_std": "mean_delta_risk_std",
            "mean_delta_risk_inst_vs_score_mean": "mean_delta_risk_inst_vs_score_mean",
        }
    )
    out_save.to_csv(output_csv, index=False)
    print(f"Wrote {output_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
