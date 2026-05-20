"""
Vectorized PromptAD Stage-2 pairwise aggregation (anomaly–normal pairs).

Definitions (Section 3.1.1 / 4): strict score comparison for z_sem, z_vis, z_fused;
pairwise_instability = Var([z_sem, z_vis]); pairwise_error = 1 - z_fused;
fused_margin = fused(anomaly) - fused(normal).
"""

from __future__ import annotations

import io
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

SETTING_KEYS = ["dataset", "category", "shot", "seed"]


def load_wide_or_long(raw_dir: Path) -> pd.DataFrame:
    wide_p = raw_dir / "unified_raw_scores_wide.csv"
    long_p = raw_dir / "unified_raw_scores_long.csv"
    if wide_p.is_file():
        return pd.read_csv(wide_p)
    if not long_p.is_file():
        raise FileNotFoundError(f"missing {wide_p} and {long_p}")
    df = pd.read_csv(long_p)
    need = {
        "sample_id",
        "condition",
        "condition_score",
        "dataset",
        "category",
        "shot",
        "seed",
        "label",
        "image_path",
    }
    if not need.issubset(df.columns):
        raise ValueError(f"long CSV missing columns: {sorted(need - set(df.columns))}")
    piv = df.pivot_table(
        index=["sample_id", "dataset", "category", "shot", "seed", "label", "image_path"],
        columns="condition",
        values="condition_score",
        aggfunc="first",
    ).reset_index()
    for c in ("semantic", "visual", "fused"):
        if c not in piv.columns:
            raise ValueError(f"pivot missing condition column {c!r}")
    piv["semantic_score"] = piv["semantic"]
    piv["visual_score"] = piv["visual"]
    piv["fused_score"] = piv["fused"]
    return piv


def build_pairs_vectorized(
    g: pd.DataFrame,
    *,
    max_pairs: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    include_image_paths: bool = False,
) -> pd.DataFrame:
    """All anomaly×normal pairs for one setting; optional uniform subsample."""
    pos = g[g["label"] == 1]
    neg = g[g["label"] == 0]
    if len(pos) == 0 or len(neg) == 0:
        return pd.DataFrame()

    sa = pos["semantic_score"].to_numpy(dtype=np.float64)[:, None]
    sn = neg["semantic_score"].to_numpy(dtype=np.float64)[None, :]
    va = pos["visual_score"].to_numpy(dtype=np.float64)[:, None]
    vn = neg["visual_score"].to_numpy(dtype=np.float64)[None, :]
    fa = pos["fused_score"].to_numpy(dtype=np.float64)[:, None]
    fn = neg["fused_score"].to_numpy(dtype=np.float64)[None, :]

    z_sem = (sa > sn).astype(np.float64)
    z_vis = (va > vn).astype(np.float64)
    z_fused = (fa > fn).astype(np.float64)
    stack = np.stack([z_sem, z_vis], axis=0)
    p_inst = np.var(stack, axis=0, ddof=0)
    margin = fa - fn

    d = str(pos["dataset"].iloc[0])
    cat = str(pos["category"].iloc[0])
    shot = int(pos["shot"].iloc[0])
    seed = int(pos["seed"].iloc[0])
    n_pos, n_neg = z_sem.shape
    flat = n_pos * n_neg

    out: Dict[str, Any] = {
        "dataset": np.full(flat, d, dtype=object),
        "category": np.full(flat, cat, dtype=object),
        "shot": np.full(flat, shot, dtype=np.int64),
        "seed": np.full(flat, seed, dtype=np.int64),
        "z_sem": z_sem.ravel(),
        "z_vis": z_vis.ravel(),
        "z_fused": z_fused.ravel(),
        "pairwise_instability": p_inst.ravel(),
        "pairwise_error": (1.0 - z_fused).ravel(),
        "fused_margin": margin.ravel(),
        "abs_fused_margin": np.abs(margin).ravel(),
    }
    if include_image_paths:
        pa = pos["image_path"].astype(str).to_numpy()
        pn = neg["image_path"].astype(str).to_numpy()
        out["image_path_anomaly"] = np.repeat(pa, n_neg)
        out["image_path_normal"] = np.tile(pn, n_pos)
        if "sample_id" in pos.columns:
            out["sample_id_anomaly"] = np.repeat(pos["sample_id"].astype(str).to_numpy(), n_neg)
            out["sample_id_normal"] = np.tile(neg["sample_id"].astype(str).to_numpy(), n_pos)

    df = pd.DataFrame(out)
    if max_pairs is not None and len(df) > max_pairs:
        if rng is None:
            rng = np.random.default_rng(42)
        idx = rng.choice(len(df), size=max_pairs, replace=False)
        df = df.iloc[idx].reset_index(drop=True)
    return df


def _process_setting_task(
    g_bytes: bytes,
    max_pairs: Optional[int],
    seed: int,
    include_image_paths: bool,
) -> Tuple[Dict[str, object], pd.DataFrame, int]:
    g = pd.read_pickle(io.BytesIO(g_bytes))
    rng = np.random.default_rng(int(seed))
    p = build_pairs_vectorized(
        g, max_pairs=max_pairs, rng=rng, include_image_paths=include_image_paths
    )
    d = str(g["dataset"].iloc[0])
    cat = str(g["category"].iloc[0])
    shot = int(g["shot"].iloc[0])
    sd = int(g["seed"].iloc[0])
    n_pos = int((g["label"] == 1).sum())
    n_neg = int((g["label"] == 0).sum())
    raw_n = n_pos * n_neg
    if len(p) == 0:
        return (
            {
                "dataset": d,
                "category": cat,
                "shot": shot,
                "seed": sd,
                "n_pairs": 0,
                "setting_auroc": float("nan"),
                "mean_pairwise_instability": float("nan"),
                "mean_pairwise_error": float("nan"),
            },
            pd.DataFrame(),
            raw_n,
        )
    return (
        {
            "dataset": d,
            "category": cat,
            "shot": shot,
            "seed": sd,
            "n_pairs": int(len(p)),
            "setting_auroc": float(p["z_fused"].mean()),
            "mean_pairwise_instability": float(p["pairwise_instability"].mean()),
            "mean_pairwise_error": float(p["pairwise_error"].mean()),
        },
        p,
        raw_n,
    )


def aggregate_pairwise_from_raw(
    df: pd.DataFrame,
    *,
    workers: int = 1,
    max_pairs_per_setting: Optional[int] = None,
    pair_sampling_seed: int = 42,
    save_pairwise: bool = True,
    include_image_paths: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Build setting-level metrics and optional full pairwise table from unified raw."""
    need = {
        "sample_id",
        "label",
        "fused_score",
        "semantic_score",
        "visual_score",
        "dataset",
        "category",
        "shot",
        "seed",
    }
    if not need.issubset(df.columns):
        raise ValueError(f"raw frame missing columns: {sorted(need - set(df.columns))}")

    tasks: List[Tuple[bytes, Optional[int], int, bool]] = []
    for _, g in df.groupby(SETTING_KEYS, sort=False):
        buf = io.BytesIO()
        g.to_pickle(buf)
        tasks.append((buf.getvalue(), max_pairs_per_setting, pair_sampling_seed, include_image_paths))

    setting_rows: List[Dict[str, object]] = []
    pair_parts: List[pd.DataFrame] = []
    total_raw = 0
    any_sub = False

    if workers <= 1:
        for g_bytes, max_pairs, seed, inc_paths in tasks:
            row, p, raw_n = _process_setting_task(g_bytes, max_pairs, seed, inc_paths)
            total_raw += raw_n
            if max_pairs is not None and raw_n > max_pairs:
                any_sub = True
            setting_rows.append(row)
            if save_pairwise and len(p):
                pair_parts.append(p)
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as ex:
            futs = [
                ex.submit(_process_setting_task, g_bytes, max_pairs, seed, inc_paths)
                for g_bytes, max_pairs, seed, inc_paths in tasks
            ]
            for fut in as_completed(futs):
                row, p, raw_n = fut.result()
                total_raw += raw_n
                if max_pairs_per_setting is not None and raw_n > max_pairs_per_setting:
                    any_sub = True
                setting_rows.append(row)
                if save_pairwise and len(p):
                    pair_parts.append(p)

    setting_df = pd.DataFrame(setting_rows)
    pairwise_df = pd.concat(pair_parts, ignore_index=True) if save_pairwise and pair_parts else pd.DataFrame()
    meta = {
        "n_settings": int(len(setting_df)),
        "n_pairs_written": int(len(pairwise_df)),
        "total_pairs_enumerated_raw": int(total_raw),
        "pair_subsampling_used": bool(any_sub),
        "max_pairs_per_setting": max_pairs_per_setting,
        "pair_sampling_seed": int(pair_sampling_seed),
        "workers": int(workers),
        "save_pairwise": bool(save_pairwise),
    }
    return setting_df, pairwise_df, meta


def margin_bucket_tertiles(abs_m: np.ndarray) -> np.ndarray:
    s = pd.Series(abs_m, dtype=float)
    if len(s) < 3:
        return np.array(["mid"] * len(s), dtype=object)
    try:
        b = pd.qcut(s, q=3, labels=["low", "mid", "high"], duplicates="drop")
        return b.astype(str).to_numpy()
    except ValueError:
        return np.array(["mid"] * len(s), dtype=object)


def controlled_margin_bucket_table(
    pairwise: pd.DataFrame,
    *,
    scope: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Tertile |fused_margin| -> margin_bucket; within each bucket, low vs high pairwise_instability
    (bottom / top third) and mean pairwise_error. scope='per_setting' or 'pooled'.
    """
    long_rows: List[Dict[str, object]] = []

    def _emit_rows(g: pd.DataFrame, key: tuple) -> None:
        abs_m = g["abs_fused_margin"].to_numpy(dtype=float)
        if len(g) < 3:
            return
        buckets = margin_bucket_tertiles(abs_m)
        g2 = g.assign(margin_bucket=buckets)
        for b in ("low", "mid", "high"):
            sub = g2[g2["margin_bucket"] == b]
            if len(sub) < 2:
                continue
            inst = sub["pairwise_instability"].to_numpy(dtype=float)
            err = sub["pairwise_error"].to_numpy(dtype=float)
            abs_sub = sub["abs_fused_margin"].to_numpy(dtype=float)
            n_sub = len(inst)
            ridx = np.argsort(inst, kind="mergesort")
            k = max(1, n_sub // 3)
            m0 = np.zeros(n_sub, dtype=bool)
            m1 = np.zeros(n_sub, dtype=bool)
            m0[ridx[:k]] = True
            for i in reversed(ridx.tolist()):
                if int(m1.sum()) >= k:
                    break
                if not m0[i]:
                    m1[i] = True
            if m0.sum() == 0 or m1.sum() == 0:
                continue
            base = {
                "margin_bucket": b,
                "n_low_I": int(m0.sum()),
                "n_high_I": int(m1.sum()),
                "error_low_I": float(np.mean(err[m0])),
                "error_high_I": float(np.mean(err[m1])),
                "mean_margin_low_I": float(np.mean(abs_sub[m0])),
                "mean_margin_high_I": float(np.mean(abs_sub[m1])),
                "mean_instability": float(np.mean(inst)),
                "error_rate": float(np.mean(err)),
            }
            if scope == "per_setting":
                d, cat, shot, seed = key
                long_rows.append(
                    {
                        "scope": "per_setting",
                        "dataset": d,
                        "category": cat,
                        "shot": int(shot),
                        "seed": int(seed),
                        **base,
                    }
                )
            else:
                long_rows.append({"scope": "pooled", "dataset": "", "category": "", "shot": -1, "seed": -1, **base})

    if scope == "per_setting":
        for key, g in pairwise.groupby(SETTING_KEYS, sort=False):
            _emit_rows(g, key)
    else:
        _emit_rows(pairwise, tuple())

    pdf = pd.DataFrame(long_rows)
    if pdf.empty:
        return pdf, pd.DataFrame()

    cm_rows: List[Dict[str, object]] = []
    for b, g in pdf.groupby("margin_bucket", sort=False):
        n0 = int(g["n_low_I"].sum())
        n1 = int(g["n_high_I"].sum())
        e0 = float(np.average(g["error_low_I"], weights=g["n_low_I"])) if n0 > 0 else float("nan")
        e1 = float(np.average(g["error_high_I"], weights=g["n_high_I"])) if n1 > 0 else float("nan")
        m0 = float(np.average(g["mean_margin_low_I"], weights=g["n_low_I"])) if n0 > 0 else float("nan")
        m1 = float(np.average(g["mean_margin_high_I"], weights=g["n_high_I"])) if n1 > 0 else float("nan")
        cm_rows.append(
            {
                "margin_bucket": b,
                "n_low_I": n0,
                "n_high_I": n1,
                "error_low_I": e0,
                "error_high_I": e1,
                "error_gap_high_minus_low": (e1 - e0) if np.isfinite(e0) and np.isfinite(e1) else float("nan"),
                "mean_margin_low_I": m0,
                "mean_margin_high_I": m1,
            }
        )
    cm = pd.DataFrame(cm_rows)
    order = {"low": 0, "mid": 1, "high": 2}
    cm["_o"] = cm["margin_bucket"].map(order)
    cm = cm.sort_values("_o").drop(columns=["_o"])
    return pdf, cm


def controlled_margin_rows(pairwise: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    long_ps, cm_ps = controlled_margin_bucket_table(pairwise, scope="per_setting")
    if not cm_ps.empty:
        return long_ps, cm_ps
    return controlled_margin_bucket_table(pairwise, scope="pooled")


def write_aggregation_done(
    out_dir: Path,
    *,
    raw_dir: Path,
    meta: Dict[str, Any],
    pairwise_path: Optional[Path] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"raw_dir": str(raw_dir), **meta}
    if pairwise_path is not None:
        payload["pairwise_metrics_csv"] = str(pairwise_path)
    (out_dir / "aggregation_done.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
