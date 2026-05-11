#!/usr/bin/env python3
"""
Section 4.3: per-setting instability--error (bucket, Spearman, error concentration).

Discovers per_sample_instability_analysis.csv under <result_root> or reads an optional
merged long CSV (setting_id + instability + error).

Usage (from PromptAD repo root):
  python utils/section_43_instability_error_analysis.py \\
    --result-root result_round1 \\
    --output-dir result_round1/section_43_instability_error
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
import re
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ------------------------------------------------------------------------------
# Column resolution
# ------------------------------------------------------------------------------

INSTABILITY_CANDIDATES: Sequence[str] = (
    "instability",
    "instability_score",
    "true_instability",
    "flip_rate",
    "I",
    "I_x",
)

ERROR_CANDIDATES: Sequence[str] = (
    "ranking_error",
    "sample_error",
    "error",
)


def _first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name in df.columns:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def resolve_instability_error_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    return _first_existing_column(df, INSTABILITY_CANDIDATES), _first_existing_column(
        df, ERROR_CANDIDATES
    )


# ------------------------------------------------------------------------------
# Path / meta (rejection_instability layout)
# ------------------------------------------------------------------------------


def discover_per_sample_csvs(result_root: str) -> List[str]:
    pattern = os.path.join(
        result_root,
        "*",
        "k_*",
        "rejection_instability",
        "CLS-*-per_sample",
        "per_sample_instability_analysis.csv",
    )
    return sorted(p for p in glob.glob(pattern) if os.path.isfile(p))


def parse_rejection_per_sample_path(csv_path: str) -> Optional[Dict[str, Any]]:
    csv_path = os.path.normpath(csv_path)
    if os.path.basename(csv_path) != "per_sample_instability_analysis.csv":
        return None
    per_sample_dir = os.path.dirname(csv_path)
    run_name = os.path.basename(per_sample_dir)  # CLS-visa-cashew-k2-seed111-per_sample
    if not run_name.endswith("-per_sample"):
        return None
    setting_id = run_name[: -len("-per_sample")]

    parent = os.path.dirname(per_sample_dir)
    if os.path.basename(parent) != "rejection_instability":
        return None
    k_dir = os.path.dirname(parent)
    m = re.fullmatch(r"k_(\d+)", os.path.basename(k_dir))
    if not m:
        return None
    k = int(m.group(1))
    dataset = os.path.basename(os.path.dirname(k_dir))
    prefix = f"CLS-{dataset}-"
    category = ""
    if run_name.startswith(prefix):
        rest = run_name[len(prefix) :]
        parts = re.split(r"-k\d+", rest, maxsplit=1)
        category = parts[0] if parts else ""
    sm = re.search(r"-seed(\d+)-", run_name)
    seed = int(sm.group(1)) if sm else None
    return {
        "setting_id": setting_id,
        "dataset": dataset,
        "category": category,
        "k": k,
        "seed": seed,
        "run_name": run_name,
        "source_csv": csv_path,
    }


# ------------------------------------------------------------------------------
# Tertile buckets: sort I ascending; three nearly equal-sized groups, remainder
# in the high bin (ranks: low=bottom, high=top).
# ------------------------------------------------------------------------------


def tertile_means(ival: np.ndarray, e: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (group_labels_0_1_2, group_mean_errors) in sort order of I (asc)."""
    order = np.argsort(ival, kind="mergesort")
    e_s = e[order]
    n = e_s.size
    n0 = n // 3
    n1 = n // 3
    g = np.zeros(n, dtype=np.int8)
    g[n0 : n0 + n1] = 1
    g[n0 + n1 :] = 2
    means = np.array(
        [
            float(np.mean(e_s[g == 0])),
            float(np.mean(e_s[g == 1])),
            float(np.mean(e_s[g == 2])),
        ],
        dtype=float,
    )
    return g, means


def round4(x: float) -> float:
    return float(f"{x:.4f}")


# ------------------------------------------------------------------------------
# DataFrame loading from discovered files
# ------------------------------------------------------------------------------


def _load_merged_input(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "setting_id" not in df.columns:
        raise ValueError("Merged CSV must contain column setting_id")
    icol, ecol = resolve_instability_error_columns(df)
    if icol is None or ecol is None:
        raise ValueError("Could not resolve instability and error columns in merged CSV")
    out = df[["setting_id", icol, ecol]].copy()
    return out.rename(columns={icol: "_I", ecol: "_E"})


def _iter_from_discovered(
    result_root: str, min_samples: int, logger: logging.Logger
) -> Tuple[List[Dict[str, Any]], List[str]]:
    paths = discover_per_sample_csvs(result_root)
    rows: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for p in paths:
        meta = parse_rejection_per_sample_path(p)
        if not meta:
            logger.warning("Skip (bad path): %s", p)
            skipped.append(p)
            continue
        try:
            df = pd.read_csv(p)
        except Exception as ex:  # noqa: BLE001
            logger.warning("Skip (read error) %s: %s", p, ex)
            skipped.append(p)
            continue
        icol, ecol = resolve_instability_error_columns(df)
        if icol is None or ecol is None:
            logger.warning("Skip (columns) %s", p)
            skipped.append(p)
            continue
        sub = df[[icol, ecol]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(sub) < min_samples:
            logger.info(
                "Skip (n=%s < min_samples=%s): %s",
                len(sub),
                min_samples,
                meta["setting_id"],
            )
            skipped.append(p)
            continue
        rows.append(
            {
                "meta": meta,
                "I": sub[icol].to_numpy(dtype=np.float64),
                "E": sub[ecol].to_numpy(dtype=np.float64),
            }
        )
    return rows, skipped


def _iter_from_merged(
    path: str, min_samples: int, logger: logging.Logger
) -> Tuple[List[Dict[str, Any]], List[str]]:
    raw = _load_merged_input(path)
    groups = raw.groupby("setting_id", sort=True)
    rows: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for setting_id, g in groups:
        sub = g[["_I", "_E"]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(sub) < min_samples:
            logger.info("Skip (merged n=%s): %s", len(sub), setting_id)
            skipped.append(str(setting_id))
            continue
        meta: Dict[str, Any] = {
            "setting_id": str(setting_id),
            "dataset": "",
            "category": "",
            "k": -1,
            "seed": None,
            "run_name": str(setting_id),
            "source_csv": path,
        }
        rows.append(
            {
                "meta": meta,
                "I": sub["_I"].to_numpy(dtype=np.float64),
                "E": sub["_E"].to_numpy(dtype=np.float64),
            }
        )
    return rows, skipped


def _build_seed_setting_id(dataset: str, category: str, k: Any, seed: Any) -> str:
    return f"CLS-{dataset}-{category}-k{int(k)}-seed{int(seed)}"


def _iter_from_per_seed_metrics(
    metrics_csv: str, min_samples: int, logger: logging.Logger
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Build sample-level (I, E) per setting using per_seed_setting_metrics.csv as index:
    - instability from .../pairwise_instability/sample_instability_table.csv (I_bin preferred)
    - ranking error from .../csv/CLS-...-per_sample_instability.csv (ranking_error_baseline preferred)
    """
    dfm = pd.read_csv(metrics_csv)
    required = {"dataset", "class", "k", "seed", "summary_path"}
    missing = required.difference(set(dfm.columns))
    if missing:
        raise ValueError(f"per-seed metrics missing columns: {sorted(missing)}")

    rows: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for _, r in dfm.iterrows():
        dataset = str(r["dataset"])
        category = str(r["class"])
        k = int(r["k"])
        seed = int(r["seed"])
        setting_id = _build_seed_setting_id(dataset, category, k, seed)
        summary_path = str(r["summary_path"])
        pair_dir = os.path.dirname(summary_path)
        data_source = str(r.get("data_source", ""))
        sub = pd.DataFrame(columns=["_I", "_E"])
        source_csv = ""

        if data_source == "result_round1":
            # Direct sample-level table with true_instability + sample_error.
            one_csv = os.path.join(
                os.path.dirname(os.path.dirname(pair_dir)),
                "rejection_instability",
                f"{setting_id}-per_sample",
                "per_sample_instability_analysis.csv",
            )
            source_csv = one_csv
            if not os.path.isfile(one_csv):
                skipped.append(setting_id)
                logger.warning("Skip missing round1 per-sample analysis: %s", setting_id)
                continue
            try:
                dfo = pd.read_csv(one_csv)
            except Exception as ex:  # noqa: BLE001
                skipped.append(setting_id)
                logger.warning("Skip read error %s: %s", setting_id, ex)
                continue
            i_col = _first_existing_column(
                dfo, ("true_instability", "instability", "instability_score", "flip_rate", "I_bin")
            )
            e_col = _first_existing_column(dfo, ("sample_error", "ranking_error", "error"))
            if i_col is None or e_col is None:
                skipped.append(setting_id)
                logger.warning("Skip round1 no I/e columns: %s", setting_id)
                continue
            sub = dfo[[i_col, e_col]].copy()
            sub = sub.rename(columns={i_col: "_I", e_col: "_E"})
            sub["_I"] = pd.to_numeric(sub["_I"], errors="coerce")
            sub["_E"] = pd.to_numeric(sub["_E"], errors="coerce")
            sub = sub.dropna()
        else:
            # Seed-search: instability in sample_instability_table + error in per_sample_instability.
            instability_csv = os.path.join(pair_dir, "sample_instability_table.csv")
            per_sample_csv = os.path.normpath(
                os.path.join(
                    pair_dir,
                    "..",
                    dataset,
                    f"k_{k}",
                    "csv",
                    f"{setting_id}-per_sample_instability.csv",
                )
            )
            source_csv = per_sample_csv
            if (not os.path.isfile(instability_csv)) or (not os.path.isfile(per_sample_csv)):
                skipped.append(setting_id)
                logger.warning("Skip missing seed-search file(s): %s", setting_id)
                continue
            try:
                dfi = pd.read_csv(instability_csv)
                dfe = pd.read_csv(per_sample_csv)
            except Exception as ex:  # noqa: BLE001
                skipped.append(setting_id)
                logger.warning("Skip read error %s: %s", setting_id, ex)
                continue

            i_col = _first_existing_column(dfi, ("I_bin", "instability", "I_cont", "flip_rate"))
            e_col = _first_existing_column(
                dfe,
                ("ranking_error_baseline", "ranking_error", "sample_error", "error"),
            )
            if i_col is None or e_col is None:
                skipped.append(setting_id)
                logger.warning("Skip seed-search no I/e columns: %s", setting_id)
                continue
            if "image_path" not in dfi.columns or "image_path" not in dfe.columns:
                skipped.append(setting_id)
                logger.warning("Skip no image_path for join: %s", setting_id)
                continue
            dfi2 = dfi[["image_path", i_col]].copy().rename(columns={i_col: "_I"})
            dfe2 = dfe[["image_path", e_col]].copy().rename(columns={e_col: "_E"})
            sub = dfi2.merge(dfe2, on="image_path", how="inner")
            sub["_I"] = pd.to_numeric(sub["_I"], errors="coerce")
            sub["_E"] = pd.to_numeric(sub["_E"], errors="coerce")
            sub = sub.dropna()

        if len(sub) < min_samples:
            skipped.append(setting_id)
            logger.info(
                "Skip %s: n=%s < min_samples=%s",
                setting_id,
                len(sub),
                min_samples,
            )
            continue

        rows.append(
            {
                "meta": {
                    "setting_id": setting_id,
                    "dataset": dataset,
                    "category": category,
                    "k": k,
                    "seed": seed,
                    "run_name": str(r.get("run_name", "")),
                    "source_csv": source_csv,
                },
                "I": sub["_I"].to_numpy(dtype=np.float64),
                "E": sub["_E"].to_numpy(dtype=np.float64),
            }
        )

    return rows, skipped


# ------------------------------------------------------------------------------
# Analyses
# ------------------------------------------------------------------------------

ALPHAS = (0.1, 0.2, 0.3)


def spearman_or_nan(ival: np.ndarray, e: np.ndarray) -> float:
    if ival.size < 2:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        r = spearmanr(ival, e, nan_policy="omit")
    try:
        corr = float(r.correlation)  # scipy 1.7+
    except AttributeError:
        corr = float(r[0])
    if corr != corr:  # nan
        return float("nan")
    return corr


def concentration_errors(ival: np.ndarray, e: np.ndarray) -> Tuple[Optional[Tuple[float, float, float]], str]:
    """If total_error>0, return (cap_0.1, cap_0.2, cap_0.3) rounded internally later."""
    n = ival.size
    total = float(np.sum(e))
    if total == 0.0:
        return None, "zero_total_error"
    order = np.argsort(-ival, kind="mergesort")
    e_desc = e[order]
    caps: List[float] = []
    for alpha in ALPHAS:
        k = max(1, int(math.ceil(alpha * n)))
        s = float(np.sum(e_desc[:k]))
        caps.append(s / total)
    return tuple(caps) if len(caps) == 3 else None, "ok"  # type: ignore[return-value]


def run_all(
    result_root: str,
    output_dir: str,
    min_samples: int,
    merged_csv: Optional[str],
    per_seed_metrics_csv: Optional[str],
    do_plots: bool,
    max_scatter_settings: int,
    scatter_seed: int,
) -> None:
    logger = logging.getLogger(__name__)
    os.makedirs(output_dir, exist_ok=True)

    if per_seed_metrics_csv and os.path.isfile(per_seed_metrics_csv):
        packs, skipped = _iter_from_per_seed_metrics(per_seed_metrics_csv, min_samples, logger)
    elif merged_csv and os.path.isfile(merged_csv):
        packs, skipped = _iter_from_merged(merged_csv, min_samples, logger)
    else:
        if merged_csv or per_seed_metrics_csv:
            logger.warning("Merged CSV not found, using discovered files under %s", result_root)
        packs, skipped = _iter_from_discovered(result_root, min_samples, logger)

    with open(os.path.join(output_dir, "skipped_or_missing_files.json"), "w", encoding="utf-8") as f:
        json.dump({"skipped": skipped, "n_skipped": len(skipped)}, f, indent=2)

    bucket_rows: List[Dict[str, Any]] = []
    corr_rows: List[Dict[str, Any]] = []
    conc_rows: List[Dict[str, Any]] = []

    for pack in packs:
        meta = pack["meta"]
        ival = pack["I"]
        e = pack["E"]
        sid = meta["setting_id"]
        n = int(ival.size)

        _g, mmeans = tertile_means(ival, e)
        bucket_rows.append(
            {
                "setting_id": sid,
                "dataset": meta.get("dataset", ""),
                "category": meta.get("category", ""),
                "k": meta.get("k", -1),
                "seed": meta.get("seed", ""),
                "n_samples": n,
                "mean_error_low": round4(mmeans[0]),
                "mean_error_mid": round4(mmeans[1]),
                "mean_error_high": round4(mmeans[2]),
            }
        )

        spe = spearman_or_nan(ival, e)
        if np.isfinite(spe):
            spe_out: Any = round4(spe)
        else:
            spe_out = spe
        corr_rows.append(
            {
                "setting_id": sid,
                "n_samples": n,
                "spearman": spe_out,
            }
        )

        caps, reason = concentration_errors(ival, e)
        if caps is None:
            conc_rows.append(
                {
                    "setting_id": sid,
                    "n_samples": n,
                    "error_captured_0.1": float("nan"),
                    "error_captured_0.2": float("nan"),
                    "error_captured_0.3": float("nan"),
                    "note": reason,
                }
            )
        else:
            conc_rows.append(
                {
                    "setting_id": sid,
                    "n_samples": n,
                    "error_captured_0.1": round4(caps[0]),
                    "error_captured_0.2": round4(caps[1]),
                    "error_captured_0.3": round4(caps[2]),
                    "note": "",
                }
            )

    df_b = pd.DataFrame(bucket_rows)
    df_c = pd.DataFrame(corr_rows)
    df_conc = pd.DataFrame(conc_rows)
    if "spearman" not in df_c.columns:
        df_c["spearman"] = pd.Series(dtype=float)
    if "error_captured_0.1" not in df_conc.columns:
        df_conc["error_captured_0.1"] = pd.Series(dtype=float)
        df_conc["error_captured_0.2"] = pd.Series(dtype=float)
        df_conc["error_captured_0.3"] = pd.Series(dtype=float)

    if df_b.empty:
        logger.error("No settings passed filters; no outputs with data.")
    df_b.to_csv(os.path.join(output_dir, "bucket_per_setting.csv"), index=False)
    df_c.to_csv(os.path.join(output_dir, "correlation_per_setting.csv"), index=False)
    df_conc.to_csv(os.path.join(output_dir, "concentration_per_setting.csv"), index=False)

    # Summaries
    n_b = len(df_b)
    bsum = {
        "n_settings": n_b,
        "n_settings_bucket": n_b,
        "mean_error_low": round4(float(np.mean(df_b["mean_error_low"]))) if n_b else float("nan"),
        "mean_error_mid": round4(float(np.mean(df_b["mean_error_mid"]))) if n_b else float("nan"),
        "mean_error_high": round4(float(np.mean(df_b["mean_error_high"]))) if n_b else float("nan"),
    }
    pd.DataFrame([bsum]).to_csv(os.path.join(output_dir, "bucket_summary.csv"), index=False)

    snum = pd.to_numeric(df_c["spearman"], errors="coerce")
    spe_vals = snum.dropna().to_numpy(dtype=np.float64)
    n_s = int(spe_vals.size)
    if n_s == 0:
        std_s = float("nan")
    elif n_s == 1:
        std_s = 0.0
    else:
        std_s = round4(float(np.std(spe_vals, ddof=1)))
    csum = {
        "n_settings_valid": n_s,
        "n_settings_total": int(len(df_c)),
        "mean_spearman": round4(float(np.mean(spe_vals))) if n_s else float("nan"),
        "median_spearman": round4(float(np.median(spe_vals))) if n_s else float("nan"),
        "std_spearman": std_s,
        "min_spearman": round4(float(np.min(spe_vals))) if n_s else float("nan"),
        "max_spearman": round4(float(np.max(spe_vals))) if n_s else float("nan"),
    }
    pd.DataFrame([csum]).to_csv(os.path.join(output_dir, "correlation_summary.csv"), index=False)

    vdf = df_conc[df_conc["error_captured_0.1"].notna()].copy()
    n_v = len(vdf)
    cons = {
        "n_settings_valid": n_v,
        "mean_error_captured_0.1": round4(float(vdf["error_captured_0.1"].mean())) if n_v else float("nan"),
        "mean_error_captured_0.2": round4(float(vdf["error_captured_0.2"].mean())) if n_v else float("nan"),
        "mean_error_captured_0.3": round4(float(vdf["error_captured_0.3"].mean())) if n_v else float("nan"),
    }
    pd.DataFrame([cons]).to_csv(os.path.join(output_dir, "concentration_summary.csv"), index=False)

    _write_paper_table_summaries(
        output_dir,
        bsum=bsum,
        cons=cons,
        n_spearman_valid=n_s,
        n_conc_valid=n_v,
        n_bucket_valid=n_b,
    )

    if do_plots and packs:
        if n_s > 0:
            _plot_spearman_hist(
                spe_vals, os.path.join(output_dir, "fig_spearman_hist.png")
            )
        if n_v > 0:
            _plot_concentration_curve_packs(
                packs, os.path.join(output_dir, "fig_concentration_curve.png")
            )
        _plot_scatter_sample(
            packs, output_dir, max_scatter_settings, scatter_seed, logger
        )
    else:
        logger.info("Skip plots (no data or do_plots=False)")

    logger.info(
        "Done. settings_with_buckets=%s, spearman_valid=%s, concentration_valid=%s, out=%s",
        n_b,
        n_s,
        n_v,
        output_dir,
    )


def _write_paper_table_summaries(
    output_dir: str,
    bsum: Dict[str, Any],
    cons: Dict[str, Any],
    n_spearman_valid: int,
    n_conc_valid: int,
    n_bucket_valid: int,
) -> None:
    """
    Text tables for direct paper use (prompt Part 1 & Part 3 final summary lines).
    """
    p = os.path.join(output_dir, "paper_table_summaries.txt")
    lo = bsum.get("mean_error_low", float("nan"))
    mi = bsum.get("mean_error_mid", float("nan"))
    hi = bsum.get("mean_error_high", float("nan"))
    c1 = cons.get("mean_error_captured_0.1", float("nan"))
    c2 = cons.get("mean_error_captured_0.2", float("nan"))
    c3 = cons.get("mean_error_captured_0.3", float("nan"))

    def _fmt(v: Any) -> str:
        if v is None or (isinstance(v, float) and v != v):
            return "nan"
        if isinstance(v, (int, np.integer)):
            return str(v)
        return f"{float(v):.4f}"

    with open(p, "w", encoding="utf-8") as f:
        f.write("# Section 4.3 — means across settings (see *_summary.csv for n_settings)\n")
        f.write(
            f"# n_settings bucket={n_bucket_valid}, spearman valid={n_spearman_valid}, "
            f"concentration valid={n_conc_valid}\n\n"
        )
        f.write("## Instability tertiles vs mean ranking error\n\n")
        f.write(f"Instability level | Low   | Mid   | High  \n")
        f.write(f"Ranking error     | {_fmt(lo):>5s} | {_fmt(mi):>5s} | {_fmt(hi):>5s}  \n\n")
        f.write("## Error concentration (mean fraction of total error in top \u03b1% of samples, by instability)\n\n")
        f.write(f"Top fraction \u03b1 | 10%   | 20%   | 30%   \n")
        f.write(f"Error captured  | {_fmt(c1):>5s} | {_fmt(c2):>5s} | {_fmt(c3):>5s}  \n")
    # Tab-separated copy for TeX/Excel
    tsv = os.path.join(output_dir, "paper_table_summaries.tsv")
    with open(tsv, "w", encoding="utf-8") as f:
        f.write("table\tcol1\tcol2\tcol3\n")
        f.write(f"bucket_error\t{_fmt(lo)}\t{_fmt(mi)}\t{_fmt(hi)}\n")
        f.write(f"concentration\t{_fmt(c1)}\t{_fmt(c2)}\t{_fmt(c3)}\n")


def _plot_spearman_hist(spe_vals: np.ndarray, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    nbin = int(min(20, max(5, spe_vals.size // 3))) if spe_vals.size else 5
    ax.hist(spe_vals, bins=nbin)
    ax.set_xlabel("Spearman corr(instability, ranking error)")
    ax.set_ylabel("Count (settings)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def concentration_capture_curve(ival: np.ndarray, e: np.ndarray, alphas: np.ndarray) -> Optional[np.ndarray]:
    n = ival.size
    total = float(np.sum(e))
    if total <= 0.0 or n < 1:
        return None
    order = np.argsort(-ival, kind="mergesort")
    e_desc = e[order]
    out = np.empty(len(alphas), dtype=np.float64)
    for j, a in enumerate(alphas):
        k = max(1, int(math.ceil(a * n)))
        out[j] = float(np.sum(e_desc[:k])) / total
    return out


def _plot_concentration_curve_packs(
    packs: List[Dict[str, Any]], out_path: str, n_grid: int = 30
) -> None:
    """Mean error_captured vs alpha, averaging per-setting curves (settings with total_error=0 excluded)."""
    alphas = np.linspace(0.02, 0.3, n_grid)
    rows: List[np.ndarray] = []
    for pack in packs:
        ival = pack["I"]
        e = pack["E"]
        c = concentration_capture_curve(ival, e, alphas)
        if c is not None:
            rows.append(c)
    if not rows:
        return
    mat = np.stack(rows, axis=0)
    mean_curve = np.mean(mat, axis=0)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(alphas, mean_curve)
    ax.set_xlabel(r"Top fraction $\alpha$")
    ax.set_ylabel("Mean error captured (across settings)")
    ax.set_xlim(0.0, 0.32)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_scatter_sample(
    packs: List[Dict[str, Any]],
    output_dir: str,
    max_scatter_settings: int,
    scatter_seed: int,
    logger: logging.Logger,
) -> None:
    rng = np.random.default_rng(scatter_seed)
    n_pack = len(packs)
    if n_pack == 0:
        return
    if n_pack <= max_scatter_settings:
        chosen = list(range(n_pack))
    else:
        chosen = list(
            np.sort(
                rng.choice(
                    np.arange(n_pack, dtype=int),
                    size=max_scatter_settings,
                    replace=False,
                )
            )
        )
    for idx in chosen:
        pack = packs[idx]
        sid = str(pack["meta"].get("setting_id", f"idx{idx}"))
        ival = pack["I"]
        e = pack["E"]
        n = int(ival.size)
        sel = np.arange(n, dtype=int)
        if n > 2000:
            sel = rng.choice(n, size=2000, replace=False)
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        ax.scatter(ival[sel], e[sel], s=3, alpha=0.35, edgecolors="none")
        ax.set_xlabel("Instability I(x)")
        ax.set_ylabel("Ranking error e(x)")
        title = sid if len(sid) <= 70 else sid[:67] + "..."
        ax.set_title(title, fontsize=7)
        fig.tight_layout()
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", sid)[:120]
        pout = os.path.join(output_dir, f"fig_scatter_{safe}.png")
        try:
            fig.savefig(pout, dpi=120)
        except OSError as ex:
            logger.warning("Could not save %s: %s", pout, ex)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sec 4.3: bucket / Spearman / error concentration (per setting)."
    )
    parser.add_argument(
        "--result-root",
        type=str,
        default="result_round1",
        help="Search root for per_sample_instability_analysis.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="result_round1/section_43_instability_error",
        help="Directory for CSVs and optional figures",
    )
    parser.add_argument("--min-samples", type=int, default=9, help="Min samples per setting")
    parser.add_argument(
        "--input-csv",
        type=str,
        default="",
        help="Optional merged long-form CSV (setting_id + I + e columns, auto-detected).",
    )
    parser.add_argument(
        "--per-seed-metrics-csv",
        type=str,
        default="",
        help="Optional index CSV (per_seed_setting_metrics.csv) to run all-seed analysis.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Do not write PNG figures (CSVs are always written).",
    )
    parser.add_argument("--max-scatter-settings", type=int, default=5)
    parser.add_argument("--scatter-seed", type=int, default=0)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()
    root = args.result_root
    if not os.path.isabs(root):
        root = os.path.join(os.path.dirname(__file__), "..", root)
        root = os.path.normpath(root)
    out = args.output_dir
    if not os.path.isabs(out):
        out = os.path.join(os.path.dirname(__file__), "..", out)
        out = os.path.normpath(out)
    logging.basicConfig(level=getattr(logging, args.log_level))
    merged = args.input_csv if args.input_csv else None
    if merged and not os.path.isabs(merged):
        merged = os.path.join(os.path.dirname(__file__), "..", merged)
        merged = os.path.normpath(merged) if merged else None
    per_seed_csv = args.per_seed_metrics_csv if args.per_seed_metrics_csv else None
    if per_seed_csv and not os.path.isabs(per_seed_csv):
        per_seed_csv = os.path.join(os.path.dirname(__file__), "..", per_seed_csv)
        per_seed_csv = os.path.normpath(per_seed_csv) if per_seed_csv else None
    run_all(
        result_root=root,
        output_dir=out,
        min_samples=args.min_samples,
        merged_csv=merged,
        per_seed_metrics_csv=per_seed_csv,
        do_plots=not args.no_plots,
        max_scatter_settings=args.max_scatter_settings,
        scatter_seed=args.scatter_seed,
    )


if __name__ == "__main__":
    main()