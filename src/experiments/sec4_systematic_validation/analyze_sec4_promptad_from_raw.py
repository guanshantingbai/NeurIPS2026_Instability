#!/usr/bin/env python3
"""
Section 4 systematic validation from PromptAD unified raw scores only.

Pairwise (anomaly–normal) definitions match Section 3.1.1: z_sem, z_vis, z_fused (strict >),
pairwise_instability = Var([z_sem, z_vis]), pairwise_error = 1 - z_fused, fused_margin.

No PromptAD train/infer and no external/PromptAD scripts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.core.selection_failure import delta_risk

SETTING_KEYS = ["dataset", "category", "shot", "seed"]


def _load_wide_or_long(raw_dir: Path) -> pd.DataFrame:
    wide_p = raw_dir / "unified_raw_scores_wide.csv"
    long_p = raw_dir / "unified_raw_scores_long.csv"
    if wide_p.is_file():
        return pd.read_csv(wide_p)
    if not long_p.is_file():
        raise FileNotFoundError(f"missing {wide_p} and {long_p}")
    df = pd.read_csv(long_p)
    need = {"sample_id", "condition", "condition_score", "dataset", "category", "shot", "seed", "label", "image_path"}
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


def _pairwise_instability(z_sem: float, z_vis: float) -> float:
    return float(np.var([z_sem, z_vis], ddof=0))


def _build_pairs_for_setting(
    g: pd.DataFrame,
    *,
    max_pairs: Optional[int],
    rng: np.random.Generator,
) -> pd.DataFrame:
    pos = g[g["label"] == 1].reset_index(drop=True)
    neg = g[g["label"] == 0].reset_index(drop=True)
    if len(pos) == 0 or len(neg) == 0:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    for i in range(len(pos)):
        for j in range(len(neg)):
            sa = float(pos.loc[i, "semantic_score"])
            sn = float(neg.loc[j, "semantic_score"])
            va = float(pos.loc[i, "visual_score"])
            vn = float(neg.loc[j, "visual_score"])
            fa = float(pos.loc[i, "fused_score"])
            fn = float(neg.loc[j, "fused_score"])
            z_sem = 1.0 if sa > sn else 0.0
            z_vis = 1.0 if va > vn else 0.0
            z_fused = 1.0 if fa > fn else 0.0
            margin = fa - fn
            rows.append(
                {
                    "dataset": pos.loc[i, "dataset"],
                    "category": pos.loc[i, "category"],
                    "shot": int(pos.loc[i, "shot"]),
                    "seed": int(pos.loc[i, "seed"]),
                    "z_sem": z_sem,
                    "z_vis": z_vis,
                    "z_fused": z_fused,
                    "pairwise_instability": _pairwise_instability(z_sem, z_vis),
                    "pairwise_error": 1.0 - z_fused,
                    "fused_margin": margin,
                    "abs_fused_margin": float(abs(margin)),
                }
            )
    out = pd.DataFrame(rows)
    if max_pairs is not None and len(out) > max_pairs:
        idx = rng.choice(len(out), size=max_pairs, replace=False)
        out = out.iloc[idx].reset_index(drop=True)
    return out


def _margin_bucket_tertiles(abs_m: np.ndarray) -> np.ndarray:
    """Labels low / mid / high by tertiles of abs margin; ties handled via pandas."""
    s = pd.Series(abs_m, dtype=float)
    if len(s) < 3:
        return np.array(["mid"] * len(s), dtype=object)
    try:
        b = pd.qcut(s, q=3, labels=["low", "mid", "high"], duplicates="drop")
        return b.astype(str).to_numpy()
    except ValueError:
        return np.array(["mid"] * len(s), dtype=object)


def _controlled_margin_bucket_table(pairwise: pd.DataFrame, *, scope: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Tertile |fused_margin| -> margin_bucket; within each bucket, low vs high pairwise_instability
    (30% / 70% quantiles) and mean pairwise_error. scope='per_setting' or 'pooled'.
    Returns (detail_long, aggregated controlled_margin_analysis).
    """
    long_rows: List[Dict[str, object]] = []
    def _emit_rows(g: pd.DataFrame, key: tuple) -> None:
        abs_m = g["abs_fused_margin"].to_numpy(dtype=float)
        if len(g) < 3:
            return
        buckets = _margin_bucket_tertiles(abs_m)
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
            if scope == "per_setting":
                d, cat, shot, seed = key
                long_rows.append(
                    {
                        "scope": "per_setting",
                        "dataset": d,
                        "category": cat,
                        "shot": int(shot),
                        "seed": int(seed),
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
                )
            else:
                long_rows.append(
                    {
                        "scope": "pooled",
                        "dataset": "",
                        "category": "",
                        "shot": -1,
                        "seed": -1,
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
                )

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


def _controlled_margin_rows(pairwise: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_ps, cm_ps = _controlled_margin_bucket_table(pairwise, scope="per_setting")
    if not cm_ps.empty:
        return long_ps, cm_ps
    long_p, cm_p = _controlled_margin_bucket_table(pairwise, scope="pooled")
    return long_p, cm_p


def _failure_conditioned(pairwise: pd.DataFrame) -> pd.DataFrame:
    fail = pairwise["pairwise_error"].to_numpy(dtype=float) > 0.5
    rows = []
    for name, col in [
        ("pairwise_instability", "pairwise_instability"),
        ("z_sem", "z_sem"),
        ("z_vis", "z_vis"),
        ("abs_fused_margin", "abs_fused_margin"),
    ]:
        v = pairwise[col].to_numpy(dtype=float)
        mf = float(np.mean(v[fail])) if fail.any() else float("nan")
        mo = float(np.mean(v[~fail])) if (~fail).any() else float("nan")
        ratio = (mf / mo) if np.isfinite(mf) and np.isfinite(mo) and mo != 0 else float("nan")
        rows.append({"signal": name, "mean_failure": mf, "mean_non_failure": mo, "ratio": ratio})
    return pd.DataFrame(rows)


def _instability_regime_failure(pairwise: pd.DataFrame) -> pd.DataFrame:
    inst = pairwise["pairwise_instability"].to_numpy(dtype=float)
    err = pairwise["pairwise_error"].to_numpy(dtype=float)
    if len(inst) < 10:
        q30, q70 = np.quantile(inst, [0.33, 0.66])
    else:
        q30, q70 = np.quantile(inst, [0.3, 0.7])
    reg = np.where(inst <= q30, "low_I", np.where(inst >= q70, "high_I", "mid_I"))
    rows = []
    for r in ("low_I", "mid_I", "high_I"):
        m = reg == r
        rows.append(
            {
                "regime": r,
                "n_pairs": int(m.sum()),
                "failure_rate": float(np.mean(err[m])) if m.any() else float("nan"),
                "mean_pairwise_instability": float(np.mean(inst[m])) if m.any() else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _near_auroc_candidates(
    setting_df: pd.DataFrame, *, epsilon: float
) -> pd.DataFrame:
    """Same (dataset, category, shot), different seed, |ΔAUROC| < epsilon."""
    rows: List[Dict[str, object]] = []
    auroc_col = "setting_auroc"
    inst_col = "mean_pairwise_instability"
    err_col = "mean_pairwise_error"
    for (d, cat, shot), g in setting_df.groupby(["dataset", "category", "shot"], sort=False):
        g = g.sort_values("seed")
        recs = g.to_dict(orient="records")
        n = len(recs)
        for i in range(n):
            for j in range(i + 1, n):
                ai = float(recs[i][auroc_col])
                aj = float(recs[j][auroc_col])
                if not (np.isfinite(ai) and np.isfinite(aj)):
                    continue
                if abs(ai - aj) >= epsilon:
                    continue
                rows.append(
                    {
                        "dataset": d,
                        "category": cat,
                        "shot": int(shot),
                        "seed_a": int(recs[i]["seed"]),
                        "seed_b": int(recs[j]["seed"]),
                        "setting_auroc_a": ai,
                        "setting_auroc_b": aj,
                        "delta_auroc": float(abs(ai - aj)),
                        "mean_pairwise_instability_a": float(recs[i][inst_col]),
                        "mean_pairwise_instability_b": float(recs[j][inst_col]),
                        "delta_mean_pairwise_instability": float(
                            abs(float(recs[i][inst_col]) - float(recs[j][inst_col]))
                        ),
                        "mean_pairwise_error_a": float(recs[i][err_col]),
                        "mean_pairwise_error_b": float(recs[j][err_col]),
                    }
                )
    return pd.DataFrame(rows)


def _decision_consequence(candidates: pd.DataFrame) -> pd.DataFrame:
    """If one always picked the higher-instability seed: excess error vs oracle min."""
    if candidates.empty:
        return pd.DataFrame()
    out_rows: List[Dict[str, object]] = []
    for _, r in candidates.iterrows():
        key_base = (r["dataset"], r["category"], int(r["shot"]))
        ia = float(r["mean_pairwise_instability_a"])
        ib = float(r["mean_pairwise_instability_b"])
        ea = float(r["mean_pairwise_error_a"])
        eb = float(r["mean_pairwise_error_b"])
        best = min(ea, eb)
        if ia >= ib:
            selected_err = ea
            higher_seed = int(r["seed_a"])
            lower_seed = int(r["seed_b"])
        else:
            selected_err = eb
            higher_seed = int(r["seed_b"])
            lower_seed = int(r["seed_a"])
        oracle = best
        dr = delta_risk(float(selected_err), float(oracle))
        out_rows.append(
            {
                "dataset": r["dataset"],
                "category": r["category"],
                "shot": int(r["shot"]),
                "seed_higher_instability": higher_seed,
                "seed_lower_instability": lower_seed,
                "mean_pairwise_error_if_pick_higher_inst": float(selected_err),
                "oracle_min_mean_pairwise_error": float(oracle),
                "delta_risk_higher_inst_vs_oracle": float(dr),
                "delta_auroc": float(r["delta_auroc"]),
            }
        )
    return pd.DataFrame(out_rows)


def _plot_controlled_margin(cm: pd.DataFrame, out: Path) -> None:
    if cm.empty:
        return
    order = ["low", "mid", "high"]
    cm = cm.set_index("margin_bucket").reindex(order).reset_index()
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    x = np.arange(len(order))
    w = 0.35
    e0 = cm["error_low_I"].to_numpy(dtype=float)
    e1 = cm["error_high_I"].to_numpy(dtype=float)
    ax.bar(x - w / 2, e0, width=w, label="low instability (pairwise)", color="#4C72B0")
    ax.bar(x + w / 2, e1, width=w, label="high instability (pairwise)", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels([o.capitalize() for o in order])
    ax.set_xlabel(r"$|\mathrm{fused\_margin}|$ tertile (within setting)")
    ax.set_ylabel("mean pairwise_error")
    ax.set_title("Sec 4: controlled margin (pairwise)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_failure_conditioned(fc: pd.DataFrame, out: Path) -> None:
    if fc.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(fc))
    ax.bar(x - 0.2, fc["mean_failure"], width=0.4, label="failure pairs", color="#C44E52")
    ax.bar(x + 0.2, fc["mean_non_failure"], width=0.4, label="non-failure pairs", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(fc["signal"].tolist(), rotation=15, ha="right")
    ax.set_ylabel("mean signal")
    ax.set_title("Sec 4: failure-conditioned signals (pairwise)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_instability_regime(reg: pd.DataFrame, out: Path) -> None:
    if reg.empty:
        return
    order = ["low_I", "mid_I", "high_I"]
    reg = reg.set_index("regime").reindex(order).reset_index()
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(reg["regime"].tolist(), reg["failure_rate"].to_numpy(dtype=float), color="#4C72B0", edgecolor="white")
    ax.set_xlabel("pairwise_instability regime (pooled tertiles)")
    ax.set_ylabel("mean pairwise_error (failure rate)")
    ax.set_title("Sec 4: instability regime failure rate")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_delta_risk(dc: pd.DataFrame, out: Path) -> None:
    if dc.empty:
        return
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    y = dc["delta_risk_higher_inst_vs_oracle"].to_numpy(dtype=float)
    ax.hist(y[np.isfinite(y)], bins=min(30, max(5, len(y) // 2)), color="steelblue", edgecolor="white")
    ax.set_xlabel("delta_risk (pick higher-mean-instability seed vs oracle min error)")
    ax.set_ylabel("count (near-AUROC pairs)")
    ax.set_title("Sec 4: decision consequence / delta risk")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=Path("outputs/cached_results/raw_scores/promptad"))
    ap.add_argument("--cache-dir", type=Path, default=Path("outputs/cached_results/sec4_systematic"))
    ap.add_argument("--fig-dir", type=Path, default=Path("outputs/figures/sec4_systematic"))
    ap.add_argument("--near-auroc-epsilon", type=float, default=0.002, help="|ΔAUROC| threshold for candidate seed pairs")
    ap.add_argument("--max-pairs-per-setting", type=int, default=None)
    ap.add_argument("--pair-sampling-seed", type=int, default=42)
    args = ap.parse_args()

    raw_dir = args.raw_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    fig_dir = args.fig_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(args.pair_sampling_seed))

    df = _load_wide_or_long(raw_dir)
    need = {"sample_id", "label", "fused_score", "semantic_score", "visual_score", "dataset", "category", "shot", "seed"}
    if not need.issubset(df.columns):
        raise SystemExit(f"raw frame missing columns: {sorted(need - set(df.columns))}")

    all_parts: List[pd.DataFrame] = []
    setting_rows: List[Dict[str, object]] = []
    total_raw = 0
    any_sub = False

    for key, g in df.groupby(SETTING_KEYS, sort=False):
        n_pos = int((g["label"] == 1).sum())
        n_neg = int((g["label"] == 0).sum())
        raw_n = n_pos * n_neg
        total_raw += raw_n
        p = _build_pairs_for_setting(g, max_pairs=args.max_pairs_per_setting, rng=rng)
        if args.max_pairs_per_setting is not None and raw_n > args.max_pairs_per_setting:
            any_sub = True
        d, cat, shot, seed = key
        if len(p) == 0:
            setting_rows.append(
                {
                    "dataset": d,
                    "category": cat,
                    "shot": int(shot),
                    "seed": int(seed),
                    "n_pairs": 0,
                    "setting_auroc": float("nan"),
                    "mean_pairwise_instability": float("nan"),
                    "mean_pairwise_error": float("nan"),
                }
            )
            continue
        setting_rows.append(
            {
                "dataset": d,
                "category": cat,
                "shot": int(shot),
                "seed": int(seed),
                "n_pairs": int(len(p)),
                "setting_auroc": float(p["z_fused"].mean()),
                "mean_pairwise_instability": float(p["pairwise_instability"].mean()),
                "mean_pairwise_error": float(p["pairwise_error"].mean()),
            }
        )
        all_parts.append(p)

    pairwise = pd.concat(all_parts, ignore_index=True) if all_parts else pd.DataFrame()
    pairwise.to_csv(cache_dir / "pairwise_metrics.csv", index=False)

    setting_df = pd.DataFrame(setting_rows)
    setting_df.to_csv(cache_dir / "setting_level_metrics.csv", index=False)

    cand = _near_auroc_candidates(setting_df, epsilon=float(args.near_auroc_epsilon))
    cand.to_csv(cache_dir / "near_auroc_candidate_pairs.csv", index=False)

    long_cm, cm = _controlled_margin_rows(pairwise)
    long_cm.to_csv(cache_dir / "controlled_margin_detail.csv", index=False)
    cm.to_csv(cache_dir / "controlled_margin_analysis.csv", index=False)

    fc = _failure_conditioned(pairwise) if len(pairwise) else pd.DataFrame()
    fc.to_csv(cache_dir / "failure_conditioned_signal_comparison.csv", index=False)

    reg = _instability_regime_failure(pairwise) if len(pairwise) else pd.DataFrame()
    reg.to_csv(cache_dir / "instability_regime_failure_rate.csv", index=False)

    dc = _decision_consequence(cand)
    dc.to_csv(cache_dir / "decision_consequence_delta_risk.csv", index=False)

    _plot_controlled_margin(cm, fig_dir / "fig_controlled_margin.png")
    _plot_failure_conditioned(fc, fig_dir / "fig_failure_conditioned_signals.png")
    _plot_instability_regime(reg, fig_dir / "fig_instability_regime_failure_rate.png")
    _plot_delta_risk(dc, fig_dir / "fig_decision_consequence_delta_risk.png")

    summary = {
        "source": "promptad_unified_raw_pairwise_only",
        "raw_dir": str(raw_dir),
        "near_auroc_epsilon": float(args.near_auroc_epsilon),
        "max_pairs_per_setting": args.max_pairs_per_setting,
        "pair_sampling_seed": int(args.pair_sampling_seed),
        "pair_subsampling_used": bool(any_sub),
        "n_settings": int(len(setting_df)),
        "n_pairs_written": int(len(pairwise)),
        "total_pairs_enumerated_raw": int(total_raw),
        "n_near_auroc_candidate_pairs": int(len(cand)),
        "controlled_margin_detail_rows": int(len(long_cm)),
        "controlled_margin_aggregated_rows": int(len(cm)),
    }
    (cache_dir / "sec4_systematic_from_raw_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (fig_dir / "from_promptad_raw_done.txt").write_text(
        "Section 4 systematic stats from PromptAD unified raw (pairwise). No external PromptAD scripts.\n",
        encoding="utf-8",
    )

    print(f"wrote {cache_dir} ({len(pairwise)} pairwise rows, {len(cand)} near-AUROC pairs)")
    print(f"figures -> {fig_dir}")


if __name__ == "__main__":
    main()
