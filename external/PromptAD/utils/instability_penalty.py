"""
Instability-aware scores for PromptAD (CLS): post-hoc penalty on fused scores.

- **Training** (``train_cls``): optional global z-score penalty
  ``metric_cal_img_with_instability_penalty`` (unchanged).
- **Inference** (``test_cls``):
  - **H2** (default): normal-centered bidirectional correction on high-``u`` test samples,
    using ``t = mean(s_final)`` over k-shot support normals.
  - **thresholded**: one-sided penalty ``s - λ u 1[u>τ]`` (legacy ablation).

Does not change the model forward pass.
"""

from __future__ import annotations

import csv
import os
import warnings
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from utils.metrics import harmonic_mean_fuse_scores, instability_aware_harmonic_fuse_scores

DEFAULT_LAMBDAS: Tuple[float, ...] = (0.0, 0.05, 0.1, 0.2, 0.5, 1.0)

_LAMBDA_SUFFIX: Dict[float, str] = {
    0.0: "0",
    0.05: "0p05",
    0.1: "0p1",
    0.2: "0p2",
    0.5: "0p5",
    1.0: "1",
}


def lambda_to_suffix(lmbda: float) -> str:
    key = float(lmbda)
    if key in _LAMBDA_SUFFIX:
        return _LAMBDA_SUFFIX[key]
    s = ("%g" % key).replace(".", "p").replace("-", "m")
    return s


def zscore(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    mu = float(np.mean(x))
    sigma = float(np.std(x))
    if sigma < eps:
        warnings.warn("zscore: std ~ 0; returning zeros", RuntimeWarning, stacklevel=2)
        return np.zeros_like(x, dtype=float)
    return (x - mu) / sigma


def compute_disagreement(
    s_sem: np.ndarray,
    s_vis: np.ndarray,
    s_final: np.ndarray,
) -> Dict[str, np.ndarray]:
    s_sem = np.asarray(s_sem, dtype=float).reshape(-1)
    s_vis = np.asarray(s_vis, dtype=float).reshape(-1)
    s_final = np.asarray(s_final, dtype=float).reshape(-1)
    n = len(s_sem)
    if len(s_vis) != n or len(s_final) != n:
        raise ValueError("s_sem, s_vis, s_final must have the same length")

    s_sem_z = zscore(s_sem)
    s_vis_z = zscore(s_vis)
    s_final_z = zscore(s_final)

    u_abs = np.abs(s_sem - s_vis)
    u_norm = np.abs(s_sem_z - s_vis_z)

    return {
        "s_sem_z": s_sem_z,
        "s_vis_z": s_vis_z,
        "s_final_z": s_final_z,
        "u_abs": u_abs,
        "u_norm": u_norm,
    }


def metric_cal_img_with_instability_penalty(
    semantic_scores: np.ndarray,
    visual_scores: np.ndarray,
    gt_list,
    lambda_penalty: float = 0.0,
) -> Dict[str, float]:
    """
    Image-level AUROC (percentage under key ``i_roc``), same contract as ``metric_cal_img``.

    Uses harmonic-mean fusion, then scores each image with::

        s = zscore(s_final) - lambda_penalty * |zscore(s_sem) - zscore(s_vis)|

    where z-scores are taken over the **current batch of eval images** (one category
    test split). ``lambda_penalty == 0`` yields the same ranking as ``s_final``,
    hence the same AUROC as ``metric_cal_img``.
    """
    s_sem = np.asarray(semantic_scores, dtype=float).reshape(-1)
    vis = np.asarray(visual_scores, dtype=float)
    if vis.ndim > 1:
        vis = vis.reshape(vis.shape[0], -1).max(axis=1)
    s_vis = vis.reshape(-1)
    s_final = harmonic_mean_fuse_scores(s_sem, s_vis)
    gt_arr = np.asarray(gt_list, dtype=int).reshape(-1)

    disc = compute_disagreement(s_sem, s_vis, s_final)
    lam = float(lambda_penalty)
    pen = apply_penalty(disc["s_final_z"], disc["u_norm"], [lam])
    img_scores = pen[lam]

    n0 = int(np.sum(gt_arr == 0))
    n1 = int(np.sum(gt_arr == 1))
    if n0 == 0 or n1 == 0:
        return {"i_roc": float("nan")}
    try:
        img_roc_auc = float(roc_auc_score(gt_arr, img_scores))
    except ValueError:
        return {"i_roc": float("nan")}
    return {"i_roc": img_roc_auc * 100.0}


def metric_cal_img_with_thresholded_instability_penalty(
    semantic_scores: np.ndarray,
    visual_scores: np.ndarray,
    gt_list,
    lambda_penalty: float = 0.1,
    quantile_q: float = 0.8,
) -> Dict[str, float]:
    """
    Image-level AUROC (percentage under ``i_roc``), inference-only Strategy 1:
    s_pen = s_final - λ * u_abs * 1[u_abs > τ], τ = quantile(u_abs, q).
    """
    s_sem = np.asarray(semantic_scores, dtype=float).reshape(-1)
    vis = np.asarray(visual_scores, dtype=float)
    if vis.ndim > 1:
        vis = vis.reshape(vis.shape[0], -1).max(axis=1)
    s_vis = vis.reshape(-1)
    s_final = harmonic_mean_fuse_scores(s_sem, s_vis)
    gt_arr = np.asarray(gt_list, dtype=int).reshape(-1)

    disc = compute_disagreement(s_sem, s_vis, s_final)
    s_pen, _, _ = apply_thresholded_penalty(
        s_final, disc["u_abs"], lambda_penalty, quantile_q
    )

    n0 = int(np.sum(gt_arr == 0))
    n1 = int(np.sum(gt_arr == 1))
    if n0 == 0 or n1 == 0:
        return {"i_roc": float("nan")}
    try:
        img_roc_auc = float(roc_auc_score(gt_arr, s_pen))
    except ValueError:
        return {"i_roc": float("nan")}
    return {"i_roc": img_roc_auc * 100.0}


def metric_cal_img_with_h2_instability_correction(
    semantic_scores: np.ndarray,
    visual_scores: np.ndarray,
    gt_list,
    t_support_normal: float,
    lambda_penalty: float = 0.1,
    quantile_q: float = 0.8,
) -> Dict[str, float]:
    """
    Image-level AUROC (``i_roc``), inference H2: bidirectional correction toward support-normal
    center ``t`` on high-``u`` samples only.
    """
    s_sem = np.asarray(semantic_scores, dtype=float).reshape(-1)
    vis = np.asarray(visual_scores, dtype=float)
    if vis.ndim > 1:
        vis = vis.reshape(vis.shape[0], -1).max(axis=1)
    s_vis = vis.reshape(-1)
    s_final = harmonic_mean_fuse_scores(s_sem, s_vis)
    gt_arr = np.asarray(gt_list, dtype=int).reshape(-1)

    disc = compute_disagreement(s_sem, s_vis, s_final)
    s_pen, _, _, _ = apply_h2_bidirectional_penalty(
        s_final,
        disc["u_abs"],
        t_support_normal,
        lambda_penalty,
        quantile_q,
    )

    n0 = int(np.sum(gt_arr == 0))
    n1 = int(np.sum(gt_arr == 1))
    if n0 == 0 or n1 == 0:
        return {"i_roc": float("nan")}
    try:
        img_roc_auc = float(roc_auc_score(gt_arr, s_pen))
    except ValueError:
        return {"i_roc": float("nan")}
    return {"i_roc": img_roc_auc * 100.0}


def instability_threshold_tau(u_abs: np.ndarray, quantile_q: float) -> float:
    """τ = quantile(u, q); e.g. q=0.8 targets the top ~20% by mass (with strict > mask)."""
    u = np.asarray(u_abs, dtype=float).reshape(-1)
    if u.size == 0:
        return float("nan")
    q = float(quantile_q)
    if not (0.0 < q < 1.0):
        raise ValueError("quantile_q must be in (0, 1)")
    return float(np.quantile(u, q))


def apply_thresholded_penalty(
    s_final: np.ndarray,
    u_abs: np.ndarray,
    lambda_penalty: float,
    quantile_q: float,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Strategy 1 (inference): s'(x) = s(x) - λ · u(x) · 1[u(x) > τ], with u = |s_sem - s_vis|,
    τ = quantile(u, q). Uses **raw** s_final and u (not z-scored).
    """
    s_final = np.asarray(s_final, dtype=float).reshape(-1)
    u_abs = np.asarray(u_abs, dtype=float).reshape(-1)
    if len(s_final) != len(u_abs):
        raise ValueError("s_final and u_abs length mismatch")
    tau = instability_threshold_tau(u_abs, quantile_q)
    mask = u_abs > tau
    s_pen = s_final - float(lambda_penalty) * u_abs * mask.astype(np.float64)
    return s_pen, tau, mask


def apply_thresholded_penalty_lambdas(
    s_final: np.ndarray,
    u_abs: np.ndarray,
    lambdas: Sequence[float],
    quantile_q: float,
) -> Tuple[Dict[float, np.ndarray], float, np.ndarray]:
    """Same τ and mask for all λ; returns dict lam -> s_pen."""
    s_final = np.asarray(s_final, dtype=float).reshape(-1)
    u_abs = np.asarray(u_abs, dtype=float).reshape(-1)
    tau = instability_threshold_tau(u_abs, quantile_q)
    mask = u_abs > tau
    out: Dict[float, np.ndarray] = {}
    for lam in lambdas:
        out[float(lam)] = s_final - float(lam) * u_abs * mask.astype(np.float64)
    return out, tau, mask


def apply_h2_bidirectional_penalty_lambdas(
    s_final: np.ndarray,
    u_abs: np.ndarray,
    t_support: float,
    lambdas: Sequence[float],
    quantile_q: float,
) -> Tuple[Dict[float, np.ndarray], float, np.ndarray, np.ndarray]:
    """
    H2 (inference): s' = s - λ u sign(s - t) 1[u > τ], with u = |s_sem - s_vis|,
    τ = quantile(u, q) on the **test** split, t = mean(s_final) on support normals.
    """
    s_final = np.asarray(s_final, dtype=float).reshape(-1)
    u_abs = np.asarray(u_abs, dtype=float).reshape(-1)
    tau = instability_threshold_tau(u_abs, quantile_q)
    mask = u_abs > tau
    dir_sign = np.sign(s_final - float(t_support))
    m = mask.astype(np.float64)
    out: Dict[float, np.ndarray] = {}
    for lam in lambdas:
        out[float(lam)] = s_final - float(lam) * u_abs * dir_sign * m
    return out, tau, mask, dir_sign


def apply_h2_bidirectional_penalty(
    s_final: np.ndarray,
    u_abs: np.ndarray,
    t_support: float,
    lambda_penalty: float,
    quantile_q: float,
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    d, tau, mask, dir_sign = apply_h2_bidirectional_penalty_lambdas(
        s_final, u_abs, t_support, [float(lambda_penalty)], quantile_q
    )
    return d[float(lambda_penalty)], tau, mask, dir_sign


def apply_penalty(
    s_final_z: np.ndarray,
    u_norm: np.ndarray,
    lambdas: Sequence[float],
) -> Dict[float, np.ndarray]:
    s_final_z = np.asarray(s_final_z, dtype=float).reshape(-1)
    u_norm = np.asarray(u_norm, dtype=float).reshape(-1)
    if len(s_final_z) != len(u_norm):
        raise ValueError("s_final_z and u_norm length mismatch")
    out: Dict[float, np.ndarray] = {}
    for lam in lambdas:
        out[float(lam)] = s_final_z - float(lam) * u_norm
    return out


def compute_ranking_error(scores: np.ndarray, y: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=int).reshape(-1)
    if len(scores) != len(y):
        raise ValueError("scores and y length mismatch")

    normal_mask = y == 0
    anomaly_mask = y == 1
    n_norm = int(normal_mask.sum())
    n_ano = int(anomaly_mask.sum())
    s_norm = scores[normal_mask]
    s_ano = scores[anomaly_mask]

    err = np.full(len(scores), np.nan, dtype=float)
    for i in range(len(scores)):
        if y[i] == 1:
            if n_norm == 0:
                err[i] = np.nan
            else:
                err[i] = float(np.mean(s_norm >= scores[i]))
        else:
            if n_ano == 0:
                err[i] = np.nan
            else:
                err[i] = float(np.mean(s_ano <= scores[i]))
    return err


def _tertile_labels(u: np.ndarray) -> np.ndarray:
    """low=0, mid=1, high=2 by 1/3 and 2/3 quantiles of u."""
    u = np.asarray(u, dtype=float).reshape(-1)
    q1, q2 = np.quantile(u, (1.0 / 3.0, 2.0 / 3.0))
    labels = np.zeros(len(u), dtype=int)
    labels[u > q1] = 1
    labels[u > q2] = 2
    return labels


def _bucket_mean_errors(
    u: np.ndarray,
    err_baseline: np.ndarray,
    err_penalty: np.ndarray,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    labels = _tertile_labels(u)
    names = ("low", "mid", "high")
    base_means: Dict[str, float] = {}
    pen_means: Dict[str, float] = {}
    for b, name in enumerate(names):
        m = labels == b
        if not np.any(m):
            base_means[name] = float("nan")
            pen_means[name] = float("nan")
        else:
            base_means[name] = float(np.nanmean(err_baseline[m]))
            pen_means[name] = float(np.nanmean(err_penalty[m]))
    return base_means, pen_means


def evaluate_thresholded_lambda_sweep(
    s_sem: np.ndarray,
    s_vis: np.ndarray,
    s_final: np.ndarray,
    y: np.ndarray,
    lambdas: Sequence[float] = DEFAULT_LAMBDAS,
    quantile_q: float = 0.8,
) -> Dict[str, object]:
    """
    Inference-time Strategy 1: penalize only samples with u_abs > quantile(u_abs, q).
    AUROC / ranking error use raw s_final vs thresholded s_pen.
    """
    s_sem = np.asarray(s_sem, dtype=float).reshape(-1)
    s_vis = np.asarray(s_vis, dtype=float).reshape(-1)
    s_final = np.asarray(s_final, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=int).reshape(-1)

    disc = compute_disagreement(s_sem, s_vis, s_final)
    u_abs = disc["u_abs"]

    penalties, tau, high_u_mask = apply_thresholded_penalty_lambdas(
        s_final, u_abs, lambdas, quantile_q
    )
    err_base = compute_ranking_error(s_final, y)

    n_class0 = int(np.sum(y == 0))
    n_class1 = int(np.sum(y == 1))
    auroc_baseline = float("nan")
    if n_class0 > 0 and n_class1 > 0:
        try:
            auroc_baseline = float(roc_auc_score(y, s_final))
        except ValueError:
            auroc_baseline = float("nan")

    # Δerror_high: mean ranking error on {u > τ} (same high-instability set as penalty mask)
    base_mean_high = float("nan")
    if np.any(high_u_mask):
        base_mean_high = float(np.nanmean(err_base[high_u_mask]))

    auroc_penalty: Dict[float, float] = {}
    mean_rank_err: Dict[float, float] = {}
    rank_err_by_lambda: Dict[float, np.ndarray] = {}
    delta_err_high: Dict[float, float] = {}
    pen_mean_high: Dict[float, float] = {}

    for lam in lambdas:
        s_pen = penalties[float(lam)]
        key = float(lam)
        if n_class0 > 0 and n_class1 > 0:
            try:
                auroc_penalty[key] = float(roc_auc_score(y, s_pen))
            except ValueError:
                auroc_penalty[key] = float("nan")
        else:
            auroc_penalty[key] = float("nan")

        err_p = compute_ranking_error(s_pen, y)
        rank_err_by_lambda[key] = err_p
        mean_rank_err[key] = float(np.nanmean(err_p))

        if np.any(high_u_mask):
            pen_mean_high[key] = float(np.nanmean(err_p[high_u_mask]))
            delta_err_high[key] = pen_mean_high[key] - base_mean_high
        else:
            pen_mean_high[key] = float("nan")
            delta_err_high[key] = float("nan")

    spearman_uabs = float("nan")
    sp_uabs_p = float("nan")
    mask_finite = np.isfinite(err_base)
    if np.sum(mask_finite) >= 2:
        r1, p1 = spearmanr(u_abs[mask_finite], err_base[mask_finite])
        spearman_uabs = float(r1) if not np.isnan(r1) else float("nan")
        sp_uabs_p = float(p1) if not np.isnan(p1) else float("nan")

    n_penalized = int(np.sum(high_u_mask))
    return {
        "disagreement": disc,
        "penalties": penalties,
        "ranking_error_baseline": err_base,
        "ranking_error_by_lambda": rank_err_by_lambda,
        "auroc_baseline": auroc_baseline,
        "auroc_penalty": auroc_penalty,
        "mean_ranking_error": mean_rank_err,
        "spearman_uabs": spearman_uabs,
        "spearman_uabs_p": sp_uabs_p,
        "delta_err_high": delta_err_high,
        "tau_u": tau,
        "high_u_mask": high_u_mask,
        "n_high_u": n_penalized,
        "quantile_q": float(quantile_q),
        "mean_rank_err_high_u_baseline": base_mean_high,
        "n_normal": n_class0,
        "n_anomaly": n_class1,
    }


def evaluate_h2_lambda_sweep(
    s_sem: np.ndarray,
    s_vis: np.ndarray,
    s_final: np.ndarray,
    y: np.ndarray,
    t_support: float,
    lambdas: Sequence[float] = DEFAULT_LAMBDAS,
    quantile_q: float = 0.8,
) -> Dict[str, object]:
    """
    H2: same reporting contract as ``evaluate_thresholded_lambda_sweep`` but penalties use
    bidirectional sign(s - t) on {u > τ}.
    """
    s_sem = np.asarray(s_sem, dtype=float).reshape(-1)
    s_vis = np.asarray(s_vis, dtype=float).reshape(-1)
    s_final = np.asarray(s_final, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=int).reshape(-1)

    disc = compute_disagreement(s_sem, s_vis, s_final)
    u_abs = disc["u_abs"]

    penalties, tau, high_u_mask, dir_sign = apply_h2_bidirectional_penalty_lambdas(
        s_final, u_abs, t_support, lambdas, quantile_q
    )
    err_base = compute_ranking_error(s_final, y)

    n_class0 = int(np.sum(y == 0))
    n_class1 = int(np.sum(y == 1))
    auroc_baseline = float("nan")
    if n_class0 > 0 and n_class1 > 0:
        try:
            auroc_baseline = float(roc_auc_score(y, s_final))
        except ValueError:
            auroc_baseline = float("nan")

    base_mean_high = float("nan")
    if np.any(high_u_mask):
        base_mean_high = float(np.nanmean(err_base[high_u_mask]))

    auroc_penalty: Dict[float, float] = {}
    mean_rank_err: Dict[float, float] = {}
    rank_err_by_lambda: Dict[float, np.ndarray] = {}
    delta_err_high: Dict[float, float] = {}
    pen_mean_high: Dict[float, float] = {}

    for lam in lambdas:
        s_pen = penalties[float(lam)]
        key = float(lam)
        if n_class0 > 0 and n_class1 > 0:
            try:
                auroc_penalty[key] = float(roc_auc_score(y, s_pen))
            except ValueError:
                auroc_penalty[key] = float("nan")
        else:
            auroc_penalty[key] = float("nan")

        err_p = compute_ranking_error(s_pen, y)
        rank_err_by_lambda[key] = err_p
        mean_rank_err[key] = float(np.nanmean(err_p))

        if np.any(high_u_mask):
            pen_mean_high[key] = float(np.nanmean(err_p[high_u_mask]))
            delta_err_high[key] = pen_mean_high[key] - base_mean_high
        else:
            pen_mean_high[key] = float("nan")
            delta_err_high[key] = float("nan")

    spearman_uabs = float("nan")
    sp_uabs_p = float("nan")
    mask_finite = np.isfinite(err_base)
    if np.sum(mask_finite) >= 2:
        r1, p1 = spearmanr(u_abs[mask_finite], err_base[mask_finite])
        spearman_uabs = float(r1) if not np.isnan(r1) else float("nan")
        sp_uabs_p = float(p1) if not np.isnan(p1) else float("nan")

    n_penalized = int(np.sum(high_u_mask))
    return {
        "disagreement": disc,
        "penalties": penalties,
        "ranking_error_baseline": err_base,
        "ranking_error_by_lambda": rank_err_by_lambda,
        "auroc_baseline": auroc_baseline,
        "auroc_penalty": auroc_penalty,
        "mean_ranking_error": mean_rank_err,
        "spearman_uabs": spearman_uabs,
        "spearman_uabs_p": sp_uabs_p,
        "delta_err_high": delta_err_high,
        "tau_u": tau,
        "high_u_mask": high_u_mask,
        "n_high_u": n_penalized,
        "quantile_q": float(quantile_q),
        "t_support_normal": float(t_support),
        "dir_sign": dir_sign,
        "mean_rank_err_high_u_baseline": base_mean_high,
        "n_normal": n_class0,
        "n_anomaly": n_class1,
    }


def evaluate_lambda_sweep(
    s_sem: np.ndarray,
    s_vis: np.ndarray,
    s_final: np.ndarray,
    y: np.ndarray,
    lambdas: Sequence[float] = DEFAULT_LAMBDAS,
) -> Dict[str, object]:
    s_sem = np.asarray(s_sem, dtype=float).reshape(-1)
    s_vis = np.asarray(s_vis, dtype=float).reshape(-1)
    s_final = np.asarray(s_final, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=int).reshape(-1)

    disc = compute_disagreement(s_sem, s_vis, s_final)
    s_final_z = disc["s_final_z"]
    u_abs = disc["u_abs"]
    u_norm = disc["u_norm"]

    penalties = apply_penalty(s_final_z, u_norm, lambdas)
    err_base = compute_ranking_error(s_final, y)

    n_class0 = int(np.sum(y == 0))
    n_class1 = int(np.sum(y == 1))
    auroc_baseline = float("nan")
    if n_class0 > 0 and n_class1 > 0:
        try:
            auroc_baseline = float(roc_auc_score(y, s_final))
        except ValueError:
            auroc_baseline = float("nan")

    base_uabs, _ = _bucket_mean_errors(u_abs, err_base, err_base)
    base_unorm, _ = _bucket_mean_errors(u_norm, err_base, err_base)
    bucket_baseline_uabs = base_uabs
    bucket_baseline_unorm = base_unorm

    auroc_penalty: Dict[float, float] = {}
    mean_rank_err: Dict[float, float] = {}
    rank_err_by_lambda: Dict[float, np.ndarray] = {}
    delta_err_high: Dict[float, float] = {}
    delta_err_high_uabs: Dict[float, float] = {}
    delta_err_high_unorm: Dict[float, float] = {}

    for lam in lambdas:
        s_pen = penalties[float(lam)]
        key = float(lam)
        if n_class0 > 0 and n_class1 > 0:
            try:
                auroc_penalty[key] = float(roc_auc_score(y, s_pen))
            except ValueError:
                auroc_penalty[key] = float("nan")
        else:
            auroc_penalty[key] = float("nan")

        err_p = compute_ranking_error(s_pen, y)
        rank_err_by_lambda[key] = err_p
        mean_rank_err[key] = float(np.nanmean(err_p))

        _, pen_uabs = _bucket_mean_errors(u_abs, err_base, err_p)
        _, pen_unorm = _bucket_mean_errors(u_norm, err_base, err_p)

        delta_err_high_uabs[key] = pen_uabs["high"] - base_uabs["high"]
        delta_err_high_unorm[key] = pen_unorm["high"] - base_unorm["high"]
        # default "high" report: u_abs buckets (per spec u = |s_sem-s_vis|)
        delta_err_high[key] = delta_err_high_uabs[key]

    spearman_uabs = float("nan")
    spearman_unorm = float("nan")
    sp_uabs_p = float("nan")
    sp_unorm_p = float("nan")
    mask_finite = np.isfinite(err_base)
    if np.sum(mask_finite) >= 2:
        r1, p1 = spearmanr(u_abs[mask_finite], err_base[mask_finite])
        r2, p2 = spearmanr(u_norm[mask_finite], err_base[mask_finite])
        spearman_uabs = float(r1) if not np.isnan(r1) else float("nan")
        spearman_unorm = float(r2) if not np.isnan(r2) else float("nan")
        sp_uabs_p = float(p1) if not np.isnan(p1) else float("nan")
        sp_unorm_p = float(p2) if not np.isnan(p2) else float("nan")

    return {
        "disagreement": disc,
        "penalties": penalties,
        "ranking_error_baseline": err_base,
        "ranking_error_by_lambda": rank_err_by_lambda,
        "auroc_baseline": auroc_baseline,
        "auroc_penalty": auroc_penalty,
        "mean_ranking_error": mean_rank_err,
        "spearman_uabs": spearman_uabs,
        "spearman_unorm": spearman_unorm,
        "spearman_uabs_p": sp_uabs_p,
        "spearman_unorm_p": sp_unorm_p,
        "bucket_baseline_uabs": bucket_baseline_uabs,
        "bucket_baseline_unorm": bucket_baseline_unorm,
        "delta_err_high_uabs": delta_err_high_uabs,
        "delta_err_high_unorm": delta_err_high_unorm,
        "delta_err_high": delta_err_high,
        "n_normal": n_class0,
        "n_anomaly": n_class1,
    }


def print_instability_summary(eval_out: Mapping[str, object], lambdas: Sequence[float] = DEFAULT_LAMBDAS) -> None:
    auroc_b = eval_out["auroc_baseline"]
    auroc_p = eval_out["auroc_penalty"]
    print("\n=== Instability penalty summary (inference-time) ===")
    print(f"Samples: normal={eval_out['n_normal']}, anomaly={eval_out['n_anomaly']}")
    print(f"AUROC baseline (s_final): {auroc_b:.6f}" if np.isfinite(auroc_b) else "AUROC baseline: nan (single-class or invalid)")
    print("AUROC penalized (s_final_z - lambda * u_norm):")
    for lam in lambdas:
        v = auroc_p[float(lam)]
        suf = lambda_to_suffix(lam)
        print(f"  lambda={lam} (s_penalty_{suf}): {v:.6f}" if np.isfinite(v) else f"  lambda={lam}: nan")

    print(
        f"Spearman(u_abs, ranking_error_baseline): rho={eval_out['spearman_uabs']:.6f}, p={eval_out['spearman_uabs_p']:.6g}"
        if np.isfinite(eval_out["spearman_uabs"])
        else "Spearman(u_abs, ...): nan"
    )
    print(
        f"Spearman(u_norm, ranking_error_baseline): rho={eval_out['spearman_unorm']:.6f}, p={eval_out['spearman_unorm_p']:.6g}"
        if np.isfinite(eval_out["spearman_unorm"])
        else "Spearman(u_norm, ...): nan"
    )

    bu = eval_out["bucket_baseline_uabs"]
    bn = eval_out["bucket_baseline_unorm"]
    print("Mean ranking error by u_abs tertile (baseline s_final):")
    for k in ("low", "mid", "high"):
        print(f"  {k}: {bu[k]:.6f}" if np.isfinite(bu[k]) else f"  {k}: nan")
    print("Mean ranking error by u_norm tertile (baseline s_final):")
    for k in ("low", "mid", "high"):
        print(f"  {k}: {bn[k]:.6f}" if np.isfinite(bn[k]) else f"  {k}: nan")

    du = eval_out["delta_err_high_uabs"]
    dn = eval_out["delta_err_high_unorm"]
    mr = eval_out["mean_ranking_error"]
    print("Delta error_high (penalty - baseline), high = u_abs tertile:")
    for lam in lambdas:
        print(f"  lambda={lam}: {du[float(lam)]:+.6f}" if np.isfinite(du[float(lam)]) else f"  lambda={lam}: nan")
    print("Delta error_high (penalty - baseline), high = u_norm tertile:")
    for lam in lambdas:
        print(f"  lambda={lam}: {dn[float(lam)]:+.6f}" if np.isfinite(dn[float(lam)]) else f"  lambda={lam}: nan")

    print("Mean ranking error (all samples, each lambda):")
    for lam in lambdas:
        print(f"  lambda={lam}: {mr[float(lam)]:.6f}" if np.isfinite(mr[float(lam)]) else f"  lambda={lam}: nan")
    print("=== End instability summary ===\n")


def print_thresholded_instability_summary(
    eval_out: Mapping[str, object],
    lambdas: Sequence[float] = DEFAULT_LAMBDAS,
) -> None:
    """Console report for Strategy 1 (thresholded u_abs penalty on raw s_final)."""
    auroc_b = eval_out["auroc_baseline"]
    auroc_p = eval_out["auroc_penalty"]
    q = float(eval_out["quantile_q"])
    tau = eval_out["tau_u"]
    print("\n=== Thresholded instability penalty (Strategy 1, inference) ===")
    print(f"Samples: normal={eval_out['n_normal']}, anomaly={eval_out['n_anomaly']}")
    print(f"quantile_q={q}  ->  tau=u_abs quantile, penalized mask: u_abs > tau")
    print(f"tau={tau:.6g}  |  n_samples penalized (u>tau)={eval_out['n_high_u']}")
    # Display AUROC as percentage (0–100), same as metric_cal_img / test_cls.
    auroc_b_pct = auroc_b * 100.0 if np.isfinite(auroc_b) else float("nan")
    print(
        f"AUROC_baseline (s_final, %): {auroc_b_pct:.4f}" if np.isfinite(auroc_b_pct) else "AUROC_baseline: nan"
    )
    print("AUROC_thresholded (s_final - lambda*u*1[u>tau], %):")
    for lam in lambdas:
        v = auroc_p[float(lam)]
        suf = lambda_to_suffix(lam)
        d = (v - auroc_b) * 100.0 if np.isfinite(v) and np.isfinite(auroc_b) else float("nan")
        vpct = v * 100.0 if np.isfinite(v) else float("nan")
        if np.isfinite(vpct) and np.isfinite(d):
            print(f"  lambda={lam} (s_penalty_{suf}): AUROC={vpct:.4f}  Delta_AUROC={d:+.4f}")
        else:
            print(f"  lambda={lam} (s_penalty_{suf}): nan")

    rho = eval_out["spearman_uabs"]
    pval = eval_out["spearman_uabs_p"]
    print(
        f"Spearman(instability=u_abs, ranking_error_baseline): rho={rho:.6f}, p={pval:.6g}"
        if np.isfinite(rho)
        else "Spearman(u_abs, ranking_error_baseline): nan"
    )

    deh = eval_out["delta_err_high"]
    print("Delta_error_high (mean err_pen - mean err_base on {u > tau}):")
    for lam in lambdas:
        x = deh[float(lam)]
        print(f"  lambda={lam}: {x:+.6f}" if np.isfinite(x) else f"  lambda={lam}: nan")
    print("=== End thresholded instability summary ===\n")


def print_h2_instability_summary(
    eval_out: Mapping[str, object],
    lambdas: Sequence[float] = DEFAULT_LAMBDAS,
) -> None:
    """Console report for H2 (normal-centered bidirectional correction)."""
    auroc_b = eval_out["auroc_baseline"]
    auroc_p = eval_out["auroc_penalty"]
    q = float(eval_out["quantile_q"])
    tau = eval_out["tau_u"]
    t_sup = eval_out["t_support_normal"]
    print("\n=== H2: normal-centered bidirectional instability correction (inference) ===")
    print(f"Samples: normal={eval_out['n_normal']}, anomaly={eval_out['n_anomaly']}")
    print(f"t_support_normal (mean s_final on k-shot normals): {t_sup:.6g}")
    print(f"quantile_q={q}  ->  tau=u_abs quantile on test, mask: u_abs > tau")
    print(f"tau={tau:.6g}  |  n_samples adjusted (u>tau)={eval_out['n_high_u']}")
    auroc_b_pct = auroc_b * 100.0 if np.isfinite(auroc_b) else float("nan")
    print(
        f"AUROC_baseline (s_final, %): {auroc_b_pct:.4f}" if np.isfinite(auroc_b_pct) else "AUROC_baseline: nan"
    )
    print("AUROC_H2 (s - lambda*u*sign(s-t)*1[u>tau], %):")
    for lam in lambdas:
        v = auroc_p[float(lam)]
        suf = lambda_to_suffix(lam)
        d = (v - auroc_b) * 100.0 if np.isfinite(v) and np.isfinite(auroc_b) else float("nan")
        vpct = v * 100.0 if np.isfinite(v) else float("nan")
        if np.isfinite(vpct) and np.isfinite(d):
            print(f"  lambda={lam} (s_penalty_{suf}): AUROC={vpct:.4f}  Delta_AUROC={d:+.4f}")
        else:
            print(f"  lambda={lam} (s_penalty_{suf}): nan")

    rho = eval_out["spearman_uabs"]
    pval = eval_out["spearman_uabs_p"]
    print(
        f"Spearman(instability=u_abs, ranking_error_baseline): rho={rho:.6f}, p={pval:.6g}"
        if np.isfinite(rho)
        else "Spearman(u_abs, ranking_error_baseline): nan"
    )

    deh = eval_out["delta_err_high"]
    print("Delta_error_high (mean err_H2 - mean err_base on {u > tau}):")
    for lam in lambdas:
        x = deh[float(lam)]
        print(f"  lambda={lam}: {x:+.6f}" if np.isfinite(x) else f"  lambda={lam}: nan")
    print("=== End H2 instability summary ===\n")


def save_instability_csv_and_summary(
    *,
    semantic_scores: np.ndarray,
    visual_score_maps: np.ndarray,
    image_paths: Sequence[str],
    labels: Sequence,
    category: str,
    save_path: str,
    lambdas: Sequence[float] = DEFAULT_LAMBDAS,
    quantile_q: float = 0.8,
    correction_mode: str = "h2",
    t_support_normal: float | None = None,
) -> str:
    s_sem = np.asarray(semantic_scores, dtype=float).reshape(-1)
    vis = np.asarray(visual_score_maps, dtype=float)
    if vis.ndim > 1:
        vis = vis.reshape(vis.shape[0], -1).max(axis=1)
    s_vis = vis.reshape(-1)
    s_final = harmonic_mean_fuse_scores(s_sem, s_vis)
    y_list = [int(np.asarray(x).item()) for x in labels]
    y = np.asarray(y_list, dtype=int).reshape(-1)

    if len(s_sem) != len(s_vis) or len(s_sem) != len(y):
        raise ValueError("length mismatch between scores and labels")
    if len(image_paths) != len(s_sem):
        raise ValueError("image_paths length mismatch")

    mode = (correction_mode or "h2").strip().lower()
    if mode == "h2":
        if t_support_normal is None or not np.isfinite(float(t_support_normal)):
            raise ValueError("correction_mode='h2' requires finite t_support_normal")
        eval_out = evaluate_h2_lambda_sweep(
            s_sem,
            s_vis,
            s_final,
            y,
            float(t_support_normal),
            lambdas=lambdas,
            quantile_q=quantile_q,
        )
    elif mode == "thresholded":
        eval_out = evaluate_thresholded_lambda_sweep(
            s_sem, s_vis, s_final, y, lambdas=lambdas, quantile_q=quantile_q
        )
    else:
        raise ValueError("correction_mode must be 'h2' or 'thresholded'")

    disc = eval_out["disagreement"]
    penalties = eval_out["penalties"]
    err_base = eval_out["ranking_error_baseline"]
    rank_by_lam = eval_out["ranking_error_by_lambda"]
    tau_u = float(eval_out["tau_u"])
    high_u_mask = np.asarray(eval_out["high_u_mask"], dtype=bool)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    header: List[str] = [
        "image_path",
        "label",
        "category",
        "s_sem",
        "s_vis",
        "s_final",
        "s_sem_z",
        "s_vis_z",
        "s_final_z",
        "u_abs",
        "u_norm",
        "tau_u",
        "high_u_mask",
        "ranking_error_baseline",
    ]
    if mode == "h2":
        header.extend(["t_support_normal", "h2_dir_sign"])
    for lam in lambdas:
        suf = lambda_to_suffix(lam)
        header.append(f"s_penalty_{suf}")
        header.append(f"ranking_error_{suf}")

    dir_sign = (
        np.asarray(eval_out["dir_sign"], dtype=float).reshape(-1)
        if mode == "h2"
        else None
    )
    t_val = float(eval_out["t_support_normal"]) if mode == "h2" else None

    rows: List[List[object]] = []
    for i in range(len(s_sem)):
        row: List[object] = [
            str(image_paths[i]),
            int(y[i]),
            str(category),
            float(s_sem[i]),
            float(s_vis[i]),
            float(s_final[i]),
            float(disc["s_sem_z"][i]),
            float(disc["s_vis_z"][i]),
            float(disc["s_final_z"][i]),
            float(disc["u_abs"][i]),
            float(disc["u_norm"][i]),
            tau_u,
            int(high_u_mask[i]),
            float(err_base[i]) if np.isfinite(err_base[i]) else "",
        ]
        if mode == "h2":
            assert t_val is not None and dir_sign is not None
            row.append(t_val)
            row.append(float(dir_sign[i]))
        for lam in lambdas:
            s_pen = penalties[float(lam)]
            err_l = rank_by_lam[float(lam)]
            row.append(float(s_pen[i]))
            row.append(float(err_l[i]) if np.isfinite(err_l[i]) else "")
        rows.append(row)

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    if mode == "h2":
        print_h2_instability_summary(eval_out, lambdas=lambdas)
    else:
        print_thresholded_instability_summary(eval_out, lambdas=lambdas)
    return save_path


def print_strategy_a_fusion_summary(
    *,
    auroc_baseline: float,
    auroc_fusion: float,
    alpha: float,
    n_normal: int,
    n_anomaly: int,
    spearman_uabs: float,
    spearman_uabs_p: float,
) -> None:
    print("\n=== Strategy A: instability-aware harmonic fusion (inference) ===")
    print(f"alpha={alpha}  |  w = 1/(1 + alpha * |s_sem - s_vis|)")
    print(f"Samples: normal={n_normal}, anomaly={n_anomaly}")
    bp = auroc_baseline * 100.0 if np.isfinite(auroc_baseline) else float("nan")
    fp = auroc_fusion * 100.0 if np.isfinite(auroc_fusion) else float("nan")
    d = fp - bp if np.isfinite(bp) and np.isfinite(fp) else float("nan")
    print(f"AUROC_baseline (harmonic s_final, %): {bp:.4f}" if np.isfinite(bp) else "AUROC_baseline: nan")
    print(f"AUROC_instability_fusion (s_new, %): {fp:.4f}" if np.isfinite(fp) else "AUROC_instability_fusion: nan")
    print(f"Delta_AUROC (fusion - baseline, % pts): {d:+.4f}" if np.isfinite(d) else "Delta_AUROC: nan")
    print(
        f"Spearman(u_abs, ranking_error_baseline): rho={spearman_uabs:.6f}, p={spearman_uabs_p:.6g}"
        if np.isfinite(spearman_uabs)
        else "Spearman(u_abs, ranking_error_baseline): nan"
    )
    print("=== End Strategy A summary ===\n")


def save_strategy_a_instability_fusion_csv(
    *,
    semantic_scores: np.ndarray,
    visual_score_maps: np.ndarray,
    image_paths: Sequence[str],
    labels: Sequence,
    category: str,
    save_path: str,
    alpha: float,
) -> str:
    """
    Export per-sample scores for Strategy A (weighted harmonic) plus ranking errors and AUROC summary.
    Pairwise / follow-up: use column ``s_final_instability_fusion`` as penalized score
    (``--penalty-col s_final_instability_fusion``).
    """
    s_sem = np.asarray(semantic_scores, dtype=float).reshape(-1)
    vis = np.asarray(visual_score_maps, dtype=float)
    if vis.ndim > 1:
        vis = vis.reshape(vis.shape[0], -1).max(axis=1)
    s_vis = vis.reshape(-1)
    y_list = [int(np.asarray(x).item()) for x in labels]
    y = np.asarray(y_list, dtype=int).reshape(-1)

    if len(s_sem) != len(s_vis) or len(s_sem) != len(y):
        raise ValueError("length mismatch between scores and labels")
    if len(image_paths) != len(s_sem):
        raise ValueError("image_paths length mismatch")

    s_final_baseline = harmonic_mean_fuse_scores(s_sem, s_vis)
    s_new = instability_aware_harmonic_fuse_scores(s_sem, s_vis, alpha)
    u_abs = np.abs(s_sem - s_vis)
    w = 1.0 / (1.0 + float(alpha) * u_abs)

    disc = compute_disagreement(s_sem, s_vis, s_final_baseline)
    err_base = compute_ranking_error(s_final_baseline, y)
    err_fus = compute_ranking_error(s_new, y)

    n0 = int(np.sum(y == 0))
    n1 = int(np.sum(y == 1))
    auroc_b = float("nan")
    auroc_f = float("nan")
    if n0 > 0 and n1 > 0:
        try:
            auroc_b = float(roc_auc_score(y, s_final_baseline))
            auroc_f = float(roc_auc_score(y, s_new))
        except ValueError:
            pass

    spearman_uabs = float("nan")
    sp_uabs_p = float("nan")
    mask_finite = np.isfinite(err_base)
    if np.sum(mask_finite) >= 2:
        r1, p1 = spearmanr(u_abs[mask_finite], err_base[mask_finite])
        spearman_uabs = float(r1) if not np.isnan(r1) else float("nan")
        sp_uabs_p = float(p1) if not np.isnan(p1) else float("nan")

    print_strategy_a_fusion_summary(
        auroc_baseline=auroc_b,
        auroc_fusion=auroc_f,
        alpha=float(alpha),
        n_normal=n0,
        n_anomaly=n1,
        spearman_uabs=spearman_uabs,
        spearman_uabs_p=sp_uabs_p,
    )

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    header = [
        "image_path",
        "label",
        "category",
        "s_sem",
        "s_vis",
        "u_abs",
        "w_instability",
        "s_final_baseline",
        "s_final_instability_fusion",
        "s_sem_z",
        "s_vis_z",
        "s_final_z",
        "u_norm",
        "ranking_error_baseline",
        "ranking_error_instability_fusion",
    ]
    rows: List[List[object]] = []
    for i in range(len(s_sem)):
        rows.append(
            [
                str(image_paths[i]),
                int(y[i]),
                str(category),
                float(s_sem[i]),
                float(s_vis[i]),
                float(u_abs[i]),
                float(w[i]),
                float(s_final_baseline[i]),
                float(s_new[i]),
                float(disc["s_sem_z"][i]),
                float(disc["s_vis_z"][i]),
                float(disc["s_final_z"][i]),
                float(disc["u_norm"][i]),
                float(err_base[i]) if np.isfinite(err_base[i]) else "",
                float(err_fus[i]) if np.isfinite(err_fus[i]) else "",
            ]
        )

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        cw = csv.writer(f)
        cw.writerow(header)
        cw.writerows(rows)

    return save_path
