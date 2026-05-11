#!/usr/bin/env python3
"""
PromptAD: instability-aware rejection analysis (standalone script).

Reads a per-sample CSV, computes one or more proxy instability measures u(x),
true pairwise branch-flip instability I(x), sample-level ranking error e(x),
correlations, plots, and risk-coverage / AUROC-coverage curves comparing
instability-based rejection, score-based rejection (harmonic rank uncertainty),
and random rejection.

Usage (from PromptAD repo root):
  python utils/rejection_instability_analysis.py \\
    --input-csv result_round1/mvtec/k_1/csv/CLS-mvtec-carpet-k1-seed111-per_sample.csv \\
    --output-dir /tmp/rejection_out

Proxies (default: all u1..u6):
  u1: |s_sem - s_vis|
  u2: |s_sem - s_vis| / (|s_sem| + |s_vis| + eps)
  u3: |z(s_sem) - z(s_vis)| within-setting z-scores
  u4: 1 / (m_local + eps), m_local = mean |s_fused(x)-s_fused(x')| over fused-score rank neighbors
  u5: u2 * u4
  u6: u2 * margin_proxy, margin from harmonic rank (high near score boundary)

  python utils/rejection_instability_analysis.py --input-csv ... --proxies u1,u2
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Callable, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "image_path",
    "image_label",
    "semantic_score",
    "visual_score",
    "harmonic_score",
]

EPS_DENOM = 1e-6
EPS_LOCAL = 1e-6
NEIGHBOR_HALF = 5  # up to 10 neighbors: 5 before + 5 after in fused-score sorted order (excluding self)

OUTPUT_COLUMNS = [
    "image_path",
    "image_label",
    "semantic_score",
    "visual_score",
    "harmonic_score",
    "proxy_instability",
    "proxy_u2",
    "proxy_u3",
    "proxy_u4",
    "proxy_u5",
    "proxy_u6",
    "true_instability",
    "sample_error",
]


# ---------------------------------------------------------------------------
# A. Load & validate
# ---------------------------------------------------------------------------


def load_and_validate_csv(csv_path: str) -> pd.DataFrame:
    """
    Load CSV and enforce schema + basic sanity.

    image_label: integer 0 = normal, 1 = anomaly.
    Also accepts string labels 'normal' / 'anomaly' if entire column uses them.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns: "
            f"{missing}\n"
            f"Present columns: {list(df.columns)}\n"
            f"Expected at least: {REQUIRED_COLUMNS}"
        )

    for c in ("semantic_score", "visual_score", "harmonic_score"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["image_path"] = df["image_path"].astype(str)

    lab = df["image_label"]
    if lab.dtype == object or str(lab.dtype).startswith("string"):
        mapping = {"normal": 0, "anomaly": 1, "Normal": 0, "Anomaly": 1}
        if lab.isin(list(mapping.keys())).all():
            df["image_label"] = lab.map(mapping)
        else:
            df["image_label"] = pd.to_numeric(lab, errors="coerce")
    else:
        df["image_label"] = pd.to_numeric(lab, errors="coerce")

    if df["image_label"].isna().any():
        bad = int(df["image_label"].isna().sum())
        raise ValueError(f"image_label has {bad} non-convertible values.")

    df["image_label"] = df["image_label"].astype(int)
    if not df["image_label"].isin([0, 1]).all():
        raise ValueError("image_label must be in {0, 1} after conversion.")

    if df[REQUIRED_COLUMNS[1:]].isna().any().any():
        raise ValueError("NaN in numeric or label columns after coercion; clean input first.")

    return df


def log_dataset_summary(df: pd.DataFrame, log: logging.Logger) -> None:
    n = len(df)
    n0 = int((df["image_label"] == 0).sum())
    n1 = int((df["image_label"] == 1).sum())
    log.info("Total samples: %d", n)
    log.info("  normal (label=0): %d", n0)
    log.info("  anomaly (label=1): %d", n1)
    if n0 == 0 or n1 == 0:
        log.warning("Single-class data: AUROC / some pairwise stats may be undefined.")


# ---------------------------------------------------------------------------
# B. Proxy functions (u1..u6)
# ---------------------------------------------------------------------------


def compute_proxy_u1(df: pd.DataFrame) -> np.ndarray:
    """u1(x) = |s_sem - s_vis| (baseline proxy)."""
    sem = df["semantic_score"].to_numpy(dtype=float)
    vis = df["visual_score"].to_numpy(dtype=float)
    return np.abs(sem - vis)


def compute_proxy_u2(df: pd.DataFrame) -> np.ndarray:
    """u2(x) = |s_sem - s_vis| / (|s_sem| + |s_vis| + eps)."""
    sem = df["semantic_score"].to_numpy(dtype=float)
    vis = df["visual_score"].to_numpy(dtype=float)
    return np.abs(sem - vis) / (np.abs(sem) + np.abs(vis) + EPS_DENOM)


def _zscore_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = float(np.mean(x))
    sig = float(np.std(x))
    if sig < 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - mu) / sig


def compute_proxy_u3(df: pd.DataFrame) -> np.ndarray:
    """u3(x) = |z(s_sem) - z(s_vis)| with z-scores within this setting (batch)."""
    sem = df["semantic_score"].to_numpy(dtype=float)
    vis = df["visual_score"].to_numpy(dtype=float)
    zs = _zscore_1d(sem)
    zv = _zscore_1d(vis)
    return np.abs(zs - zv)


def compute_proxy_u4(df: pd.DataFrame, k_half: int = NEIGHBOR_HALF) -> np.ndarray:
    """
    u4(x) = 1 / (m_local(x) + eps).

    Sort by s_fused (harmonic_score). For each sample (by row index), take rank neighbors:
    up to k_half positions before and k_half after in sorted order (excluding self),
    m_local = mean |s_fused(x) - s_fused(x')| over those neighbors. Boundary: fewer neighbors.
    """
    h = df["harmonic_score"].to_numpy(dtype=float)
    n = len(h)
    order = np.argsort(h, kind="mergesort")
    m_local = np.full(n, np.nan, dtype=float)

    for pos in range(n):
        lo = max(0, pos - k_half)
        hi = min(n - 1, pos + k_half)
        neigh_pos = [p for p in range(lo, hi + 1) if p != pos]
        idx_self = int(order[pos])
        if not neigh_pos:
            m_local[idx_self] = np.nan
            continue
        diffs = [abs(h[idx_self] - h[int(order[p])]) for p in neigh_pos]
        m_local[idx_self] = float(np.mean(diffs))

    return 1.0 / (m_local + EPS_LOCAL)


def compute_proxy_u5(df: pd.DataFrame) -> np.ndarray:
    """u5(x) = u2(x) * u4(x)."""
    return compute_proxy_u2(df) * compute_proxy_u4(df)


def compute_score_based_proxy(df: pd.DataFrame) -> np.ndarray:
    """
    Score-based uncertainty from harmonic_score within this setting.

    rank = rankdata(harmonic_score, method='average') / N in (0, 1],
    u_score(x) = 1 - 2 * |rank(x) - 0.5|  (high near decision boundary / mid-rank).
    Same vector interface as compute_proxy_u1(df): one row per sample.
    """
    h = df["harmonic_score"].to_numpy(dtype=float)
    n = int(h.size)
    if n == 0:
        return np.zeros(0, dtype=float)
    rk = rankdata(h, method="average")
    rank = rk / float(n)
    return 1.0 - 2.0 * np.abs(rank - 0.5)


def compute_proxy_u6(df: pd.DataFrame) -> np.ndarray:
    """
    u6(x) = u2(x) * margin_proxy(x).

    margin_proxy uses harmonic_score fused rank (same construction as score-based
    uncertainty): rank = rankdata(s_fused, method='average') / N,
    margin_proxy = 1 - 2 * |rank - 0.5| (large near mid-rank / decision boundary).
    """
    return compute_proxy_u2(df) * compute_score_based_proxy(df)


PROXY_REGISTRY: Dict[str, Callable[[pd.DataFrame], np.ndarray]] = {
    "u1": compute_proxy_u1,
    "u2": compute_proxy_u2,
    "u3": compute_proxy_u3,
    "u4": compute_proxy_u4,
    "u5": compute_proxy_u5,
    "u6": compute_proxy_u6,
}


# ---------------------------------------------------------------------------
# C. True instability & sample error
# ---------------------------------------------------------------------------


def compute_true_instability_and_sample_error(
    df: pd.DataFrame,
    log: logging.Logger,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each sample x, iterate all opposite-label samples x'.

    Convention: anomaly should score **higher** than normal (AD higher-is-anomaly).

    True instability I(x) (branch flip rate over opposite-class pairs):
      - If x is **anomaly** and x' is **normal**:
            z_sem = 1[ semantic(x) > semantic(x') ]
            z_vis = 1[ visual(x) > visual(x') ]
            flip = 1[ z_sem != z_vis ]

      - If x is **normal** and x' is **anomaly**:
            z_sem = 1[ semantic(x') > semantic(x) ]
            z_vis = 1[ visual(x') > visual(x) ]
            flip = 1[ z_sem != z_vis ]

      I(x) = mean flip over all opposite-class partners.

    Sample error e(x) (harmonic_score as final score):
      - If x is **anomaly**: wrong if harmonic(x) <= harmonic(x') for normal x'.
      - If x is **normal**: wrong if harmonic(x) >= harmonic(x') for anomaly x'.

      e(x) = mean wrong over all opposite-class partners.
    """
    n = len(df)
    lab = df["image_label"].to_numpy(dtype=np.int32)
    sem = df["semantic_score"].to_numpy(dtype=float)
    vis = df["visual_score"].to_numpy(dtype=float)
    h = df["harmonic_score"].to_numpy(dtype=float)

    idx_norm = np.where(lab == 0)[0]
    idx_anom = np.where(lab == 1)[0]

    I = np.zeros(n, dtype=float)
    e = np.zeros(n, dtype=float)

    if len(idx_norm) == 0 or len(idx_anom) == 0:
        log.warning("Missing one class; filling I and e with nan.")
        return np.full(n, np.nan), np.full(n, np.nan)

    for i in range(n):
        partners = idx_norm if lab[i] == 1 else idx_anom

        flips: List[float] = []
        errs: List[float] = []

        for j in partners:
            if lab[i] == 1:
                z_sem = 1.0 if sem[i] > sem[j] else 0.0
                z_vis = 1.0 if vis[i] > vis[j] else 0.0
                wrong = 1.0 if h[i] <= h[j] else 0.0
            else:
                z_sem = 1.0 if sem[j] > sem[i] else 0.0
                z_vis = 1.0 if vis[j] > vis[i] else 0.0
                wrong = 1.0 if h[i] >= h[j] else 0.0

            flips.append(1.0 if z_sem != z_vis else 0.0)
            errs.append(wrong)

        I[i] = float(np.mean(flips))
        e[i] = float(np.mean(errs))

    return I, e


def build_analysis_frame(df: pd.DataFrame, log: logging.Logger) -> pd.DataFrame:
    """Attach u1..u6 (u1 duplicated as proxy_instability), true_instability, sample_error."""
    out = df[REQUIRED_COLUMNS].copy()
    log.info("Computing proxies u1..u6 ...")
    out["proxy_instability"] = compute_proxy_u1(df)
    out["proxy_u2"] = compute_proxy_u2(df)
    out["proxy_u3"] = compute_proxy_u3(df)
    out["proxy_u4"] = compute_proxy_u4(df)
    out["proxy_u5"] = compute_proxy_u5(df)
    out["proxy_u6"] = compute_proxy_u6(df)
    log.info("Computing true_instability I(x) and sample_error e(x) ...")
    I, err = compute_true_instability_and_sample_error(df, log)
    out["true_instability"] = I
    out["sample_error"] = err
    return out


# ---------------------------------------------------------------------------
# D. Save
# ---------------------------------------------------------------------------


def save_per_sample_results(df: pd.DataFrame, path: str, log: logging.Logger) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    df[OUTPUT_COLUMNS].to_csv(path, index=False)
    log.info("Wrote per-sample analysis CSV: %s", path)


# ---------------------------------------------------------------------------
# E. Stats & plots
# ---------------------------------------------------------------------------


def _finite_spearman(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """Return (rho, p); nan if too few points or either series is constant."""
    m = np.isfinite(a) & np.isfinite(b)
    aa = a[m]
    bb = b[m]
    if aa.size < 3:
        return float("nan"), float("nan")
    if len(np.unique(aa)) < 2 or len(np.unique(bb)) < 2:
        return float("nan"), float("nan")
    r, p = spearmanr(aa, bb)
    r = float(r) if r == r else float("nan")
    p = float(p) if p == p else float("nan")
    return r, p


def assign_equal_frequency_tertiles(values: np.ndarray) -> np.ndarray:
    """Split finite values into three nearly equal-count groups by sorting u."""
    values = np.asarray(values, dtype=float)
    n = int(values.size)
    out = np.full(n, "", dtype=object)
    m = np.isfinite(values)
    idx = np.flatnonzero(m)
    if idx.size == 0:
        return out
    vv = values[idx]
    order = np.argsort(vv, kind="mergesort")
    sorted_idx = idx[order]
    k = sorted_idx.size
    t1 = k // 3
    t2 = (2 * k) // 3
    out[sorted_idx[:t1]] = "low"
    out[sorted_idx[t1:t2]] = "mid"
    out[sorted_idx[t2:]] = "high"
    return out


def print_and_plot_correlations(
    df: pd.DataFrame,
    out_dir: str,
    log: logging.Logger,
    proxy_name: str,
    proxy_values: np.ndarray,
    file_prefix: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    u = np.asarray(proxy_values, dtype=float)
    I = df["true_instability"].to_numpy(float)
    e = df["sample_error"].to_numpy(float)

    r_ui, p_ui = _finite_spearman(u, I)
    r_ue, p_ue = _finite_spearman(u, e)
    r_ie, p_ie = _finite_spearman(I, e)

    log.info(
        "[%s] Spearman(proxy, true_instability): rho=%.4f p=%.4g",
        proxy_name,
        r_ui,
        p_ui,
    )
    log.info("[%s] Spearman(proxy, sample_error):     rho=%.4f p=%.4g", proxy_name, r_ue, p_ue)
    log.info("[%s] Spearman(true_instability, sample_error): rho=%.4f p=%.4g", proxy_name, r_ie, p_ie)

    def scatter(x: np.ndarray, y: np.ndarray, xl: str, yl: str, fname: str) -> None:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(x, y, alpha=0.35, s=12, edgecolors="none")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        rho, _ = _finite_spearman(x, y)
        ax.set_title(f"{proxy_name}  Spearman rho={rho:.3f}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fp = os.path.join(out_dir, f"{file_prefix}{fname}")
        fig.savefig(fp, dpi=150)
        plt.close(fig)
        log.info("Saved figure: %s", fp)

    scatter(u, I, f"proxy {proxy_name}", "true_instability I(x)", "scatter_proxy_vs_true.png")
    scatter(u, e, f"proxy {proxy_name}", "sample_error e(x)", "scatter_proxy_vs_error.png")
    scatter(I, e, "true_instability I(x)", "sample_error e(x)", "scatter_true_vs_error.png")

    bucket = assign_equal_frequency_tertiles(u)
    use = bucket != ""
    dfb = pd.DataFrame({"I": I[use], "e": e[use], "bucket": bucket[use]})
    agg = dfb.groupby("bucket", observed=True).agg(mean_I=("I", "mean"), mean_e=("e", "mean"))
    for b in ("low", "mid", "high"):
        if b not in agg.index:
            agg.loc[b] = (np.nan, np.nan)
    agg = agg.reindex(["low", "mid", "high"])

    log.info("[%s] Mean true_instability / sample_error by proxy tertile:\n%s", proxy_name, agg.to_string())

    fig, ax = plt.subplots(figsize=(7, 4))
    xpos = np.arange(len(agg))
    w = 0.35
    ax.bar(xpos - w / 2, agg["mean_I"], width=w, label="mean true_instability")
    ax.bar(xpos + w / 2, agg["mean_e"], width=w, label="mean sample_error")
    ax.set_xticks(xpos)
    ax.set_xticklabels(list(agg.index))
    ax.set_xlabel(f"proxy {proxy_name} tertile (equal-frequency)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fp = os.path.join(out_dir, f"{file_prefix}bar_tertile_proxy.png")
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    log.info("Saved figure: %s", fp)


# ---------------------------------------------------------------------------
# F. Hard rejection vs random (reusable)
# ---------------------------------------------------------------------------


def _auroc_safe(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y, dtype=int).reshape(-1)
    s = np.asarray(s, dtype=float).reshape(-1)
    m = np.isfinite(s)
    y, s = y[m], s[m]
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y, s))
    except ValueError:
        return float("nan")


def _deterministic_rejection_curve(
    df: pd.DataFrame,
    order: np.ndarray,
    coverage_grid: np.ndarray,
) -> pd.DataFrame:
    """Keep samples in `order` prefix (ascending proxy / low uncertainty first)."""
    n = len(df)
    y = df["image_label"].to_numpy(dtype=int)
    h = df["harmonic_score"].to_numpy(dtype=float)
    err = df["sample_error"].to_numpy(dtype=float)
    rows: List[Dict[str, float]] = []
    for cov in coverage_grid:
        k = max(1, int(np.floor(float(cov) * n)))
        acc_idx = order[:k]
        m_err = float(np.mean(err[acc_idx]))
        auc = _auroc_safe(y[acc_idx], h[acc_idx])
        rows.append({"coverage": float(cov), "k": k, "mean_error": m_err, "auroc": auc})
    return pd.DataFrame(rows)


def _random_rejection_curve(
    df: pd.DataFrame,
    coverage_grid: np.ndarray,
    n_random: int,
    seed: int,
) -> pd.DataFrame:
    n = len(df)
    y = df["image_label"].to_numpy(dtype=int)
    h = df["harmonic_score"].to_numpy(dtype=float)
    err = df["sample_error"].to_numpy(dtype=float)
    rows_rand: List[Dict[str, float]] = []
    rng = np.random.default_rng(seed)
    for cov in coverage_grid:
        k = max(1, int(np.floor(float(cov) * n)))
        rand_errs: List[float] = []
        rand_aucs: List[float] = []
        for _ in range(n_random):
            sub = rng.choice(n, size=k, replace=False)
            rand_errs.append(float(np.mean(err[sub])))
            rand_aucs.append(_auroc_safe(y[sub], h[sub]))
        rows_rand.append(
            {
                "coverage": float(cov),
                "k": k,
                "mean_error": float(np.nanmean(rand_errs)),
                "auroc": float(np.nanmean(rand_aucs)),
            }
        )
    return pd.DataFrame(rows_rand)


def run_rejection_experiment(
    df: pd.DataFrame,
    proxy_values: np.ndarray,
    u_score: np.ndarray,
    out_dir: str,
    log: logging.Logger,
    coverage_grid: np.ndarray,
    n_random: int,
    seed: int,
    file_prefix: str = "",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Instability-based: sort by ascending instability proxy (NaN -> +inf, dropped first).

    Score-based: same rule on u_score (harmonic rank uncertainty).

    Random: same k, uniform subset, mean over n_random repeats.

    Plots and CSVs: instability, score, random on shared figures.
    """
    u = np.asarray(proxy_values, dtype=float)
    u_sort = np.where(np.isfinite(u), u, np.inf)
    order_inst = np.argsort(u_sort, kind="mergesort")

    us = np.asarray(u_score, dtype=float)
    us_sort = np.where(np.isfinite(us), us, np.inf)
    order_score = np.argsort(us_sort, kind="mergesort")

    dfi = _deterministic_rejection_curve(df, order_inst, coverage_grid)
    dfs = _deterministic_rejection_curve(df, order_score, coverage_grid)
    dfr = _random_rejection_curve(df, coverage_grid, n_random, seed)

    dfi.to_csv(os.path.join(out_dir, f"{file_prefix}rejection_instability_curve.csv"), index=False)
    dfs.to_csv(os.path.join(out_dir, f"{file_prefix}rejection_score_curve.csv"), index=False)
    dfr.to_csv(os.path.join(out_dir, f"{file_prefix}rejection_random_curve.csv"), index=False)
    log.info("Wrote %srejection_*_curve.csv (instability, score, random)", file_prefix)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(dfi["coverage"], dfi["mean_error"], marker="o", label="instability-based")
    ax.plot(dfs["coverage"], dfs["mean_error"], marker="^", label="score-based")
    ax.plot(dfr["coverage"], dfr["mean_error"], marker="s", label=f"random (mean of {n_random})")
    ax.set_xlabel("coverage (fraction kept, low proxy first)")
    ax.set_ylabel("mean sample_error on accepted set")
    ax.invert_xaxis()
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{file_prefix}curve_risk_coverage.png"), dpi=150)
    plt.close(fig)
    log.info("Saved %scurve_risk_coverage.png", file_prefix)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(dfi["coverage"], dfi["auroc"], marker="o", label="instability-based")
    ax.plot(dfs["coverage"], dfs["auroc"], marker="^", label="score-based")
    ax.plot(dfr["coverage"], dfr["auroc"], marker="s", label=f"random (mean of {n_random})")
    ax.set_xlabel("coverage (fraction kept)")
    ax.set_ylabel("AUROC (harmonic_score) on accepted set")
    ax.invert_xaxis()
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{file_prefix}curve_auroc_coverage.png"), dpi=150)
    plt.close(fig)
    log.info("Saved %scurve_auroc_coverage.png", file_prefix)

    return dfi, dfs, dfr


def summarize_curve_vs_random(dfx: pd.DataFrame, dfr: pd.DataFrame) -> Dict[str, float]:
    """Mean error / AUROC of curve `dfx` minus random baseline (aligned on coverage)."""
    merged = dfx.merge(dfr, on="coverage", suffixes=("_x", "_rand"))
    delta_risk = merged["mean_error_x"] - merged["mean_error_rand"]
    delta_auroc = merged["auroc_x"] - merged["auroc_rand"]
    wins = (merged["mean_error_x"] < merged["mean_error_rand"]).astype(float)
    return {
        "mean_delta_risk": float(np.nanmean(delta_risk.to_numpy())),
        "win_rate_vs_random": float(np.nanmean(wins.to_numpy())),
        "mean_delta_auroc": float(np.nanmean(delta_auroc.to_numpy())),
    }


def summarize_proxy_vs_random(dfi: pd.DataFrame, dfr: pd.DataFrame) -> Dict[str, float]:
    """Instability vs random (same merge keys as legacy column names)."""
    merged = dfi.merge(dfr, on="coverage", suffixes=("_inst", "_rand"))
    delta_risk = merged["mean_error_inst"] - merged["mean_error_rand"]
    delta_auroc = merged["auroc_inst"] - merged["auroc_rand"]
    wins = (merged["mean_error_inst"] < merged["mean_error_rand"]).astype(float)
    return {
        "mean_delta_risk": float(np.nanmean(delta_risk.to_numpy())),
        "win_rate_vs_random": float(np.nanmean(wins.to_numpy())),
        "mean_delta_auroc": float(np.nanmean(delta_auroc.to_numpy())),
    }


def summarize_inst_vs_score(dfi: pd.DataFrame, dfs: pd.DataFrame) -> Dict[str, float]:
    """Instability minus score-based on mean_error (negative => instability wins)."""
    merged = dfi.merge(dfs, on="coverage", suffixes=("_inst", "_score"))
    delta_risk = merged["mean_error_inst"] - merged["mean_error_score"]
    return {"mean_delta_risk_inst_vs_score": float(np.nanmean(delta_risk.to_numpy()))}


# ---------------------------------------------------------------------------
# G. CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Instability-aware rejection analysis for PromptAD per-sample CSVs.")
    p.add_argument("--input-csv", type=str, required=True, help="Path to per-sample CSV.")
    p.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output directory (default: <input_csv_dir>/rejection_analysis_out)",
    )
    p.add_argument(
        "--proxies",
        type=str,
        default="u1,u2,u3,u4,u5,u6",
        help="Comma-separated subset of proxies to run, e.g. u1,u3,u6",
    )
    p.add_argument("--random-repeats", type=int, default=20, help="Random rejection Monte Carlo repeats.")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for random rejection.")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    log = logging.getLogger("rejection_instability")

    args = parse_args()
    input_csv = os.path.abspath(args.input_csv)
    setting_name = os.path.basename(input_csv)
    if args.output_dir.strip():
        out_dir = os.path.abspath(args.output_dir.strip())
    else:
        out_dir = os.path.join(os.path.dirname(input_csv), "rejection_analysis_out")

    os.makedirs(out_dir, exist_ok=True)
    log.info("Setting (input basename): %s", setting_name)
    log.info("Input CSV: %s", input_csv)
    log.info("Output dir: %s", out_dir)

    proxy_keys = [x.strip() for x in args.proxies.split(",") if x.strip()]
    for pk in proxy_keys:
        if pk not in PROXY_REGISTRY:
            raise ValueError(f"Unknown proxy {pk!r}. Choose from {list(PROXY_REGISTRY.keys())}")

    log.info("[A] Loading and validating ...")
    df = load_and_validate_csv(input_csv)
    log_dataset_summary(df, log)

    log.info("[B] Computing per-sample metrics (all proxies + I + e) ...")
    analyzed = build_analysis_frame(df, log)
    u_score = compute_score_based_proxy(analyzed)

    out_csv = os.path.join(out_dir, "per_sample_instability_analysis.csv")
    log.info("[C] Saving per-sample table ...")
    save_per_sample_results(analyzed, out_csv, log)

    cov = np.round(np.arange(1.0, 0.45, -0.05), 2)
    cov = np.clip(cov, 0.5, 1.0)

    comparison_rows: List[Dict[str, object]] = []

    for pk in proxy_keys:
        log.info("======== Proxy=%s | setting=%s ========", pk, setting_name)
        col_map = {
            "u1": "proxy_instability",
            "u2": "proxy_u2",
            "u3": "proxy_u3",
            "u4": "proxy_u4",
            "u5": "proxy_u5",
            "u6": "proxy_u6",
        }
        u = analyzed[col_map[pk]].to_numpy(dtype=float)
        I = analyzed["true_instability"].to_numpy(dtype=float)
        e = analyzed["sample_error"].to_numpy(dtype=float)

        r_it, _ = _finite_spearman(u, I)
        r_ie, _ = _finite_spearman(u, e)

        proxy_subdir = os.path.join(out_dir, pk)
        os.makedirs(proxy_subdir, exist_ok=True)
        prefix = f"{pk}_"

        log.info("[D] Correlations and plots for %s ...", pk)
        print_and_plot_correlations(analyzed, proxy_subdir, log, pk, u, prefix)

        log.info("[E] Hard rejection experiment for %s ...", pk)
        dfi, dfs, dfr = run_rejection_experiment(
            analyzed,
            u,
            u_score,
            proxy_subdir,
            log,
            coverage_grid=cov,
            n_random=args.random_repeats,
            seed=args.seed,
            file_prefix=prefix,
        )
        summ_ir = summarize_proxy_vs_random(dfi, dfr)
        summ_sr = summarize_curve_vs_random(dfs, dfr)
        summ_is = summarize_inst_vs_score(dfi, dfs)
        log.info(
            "Setting %s | proxy=%s: inst_vs_random mean_delta_risk=%.6f | "
            "score_vs_random mean_delta_risk=%.6f | inst_vs_score mean_delta_risk=%.6f",
            setting_name,
            pk,
            summ_ir["mean_delta_risk"],
            summ_sr["mean_delta_risk"],
            summ_is["mean_delta_risk_inst_vs_score"],
        )
        comparison_rows.append(
            {
                "proxy_name": pk,
                "spearman_proxy_true": r_it,
                "spearman_proxy_error": r_ie,
                "mean_delta_risk": summ_ir["mean_delta_risk"],
                "win_rate_vs_random": summ_ir["win_rate_vs_random"],
                "mean_delta_auroc": summ_ir["mean_delta_auroc"],
                "mean_delta_risk_score": summ_sr["mean_delta_risk"],
                "win_rate_score_vs_random": summ_sr["win_rate_vs_random"],
                "mean_delta_risk_inst_vs_score": summ_is["mean_delta_risk_inst_vs_score"],
            }
        )

    comp_path = os.path.join(out_dir, "proxy_comparison.csv")
    pd.DataFrame(comparison_rows).to_csv(comp_path, index=False)
    log.info("Wrote proxy_comparison.csv: %s", comp_path)

    log.info("Done.")


if __name__ == "__main__":
    main()
