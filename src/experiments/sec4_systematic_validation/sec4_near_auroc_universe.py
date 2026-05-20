#!/usr/bin/env python3
"""
Near-AUROC ambiguity universe for Section 4 paper figures.

Statistical universe (per coverage c, epsilon ε):

  C(ε) = { (A, B) : same (dataset, category, shot), |AUROC_A − AUROC_B| < ε }

Roles (per pair, tie → smaller seed id):
  - AUROC-selected: higher setting AUROC
  - instability-selected: lower mean_pairwise_instability

Failure (aligned with mechanism ``failure_s2`` in ``mechanism_driven_analysis.py``):

  failure ⇔ (risk_AUROC − risk_instability) > τ

Default τ = 0.01 (``delta_risk > 0.01`` in upstream mechanism analysis).
Note: ``failure_driven_analysis.py`` uses a *different* label (``gap_auc > τ``);
figures here follow the user-specified Δrisk definition.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# Paper / upstream defaults (audit before changing).
DEFAULT_EPSILON = 0.002
DEFAULT_TAU = 0.01
DEFAULT_EPSILONS_FIG7 = (0.002, 0.005)
DEFAULT_COVERAGES_FIG7 = (0.9, 0.8, 0.7, 0.5, 0.3)
DEFAULT_COVERAGES_FIG56_REGIME = (0.5, 0.7, 0.8)
SIGNAL_ORDER = ["instability", "gap", "score_var", "margin"]
REGIME_ORDER = ["low_I", "mid_I", "high_I"]
BUCKET_ORDER = ["low", "mid", "high"]
BUCKET_LABELS = ["Low", "Mid", "High"]


def _risk_at_coverage(rc: pd.DataFrame, cov: float) -> float:
    c = rc["coverage"].astype(float).to_numpy()
    r = rc["mean_pairwise_risk"].astype(float).to_numpy()
    m = np.isfinite(r)
    if not np.any(m):
        return float("nan")
    c, r = c[m], r[m]
    order = np.argsort(c)
    c, r = c[order], r[order]
    return float(np.interp(float(cov), c, r, left=float(r[0]), right=float(r[-1])))


def build_risk_lookup(rc: pd.DataFrame) -> Dict[Tuple[str, str, int, int], pd.DataFrame]:
    out: Dict[Tuple[str, str, int, int], pd.DataFrame] = {}
    for key, g in rc.groupby(["dataset", "category", "shot", "seed"], sort=False):
        d, cat, shot, sd = key
        sub = g[["coverage", "mean_pairwise_risk"]].copy()
        sub["coverage"] = sub["coverage"].astype(float)
        sub = sub.sort_values("coverage").drop_duplicates("coverage", keep="last")
        out[(str(d), str(cat), int(shot), int(sd))] = sub.reset_index(drop=True)
    return out


def assign_roles(
    seed_a: int,
    seed_b: int,
    auroc_a: float,
    auroc_b: float,
    inst_a: float,
    inst_b: float,
) -> Tuple[int, int, float, float, float, float]:
    """Return seed_auc, seed_inst, auroc_auc, auroc_inst, inst_auc, inst_inst."""
    if auroc_a > auroc_b or (np.isclose(auroc_a, auroc_b, rtol=0, atol=1e-12) and seed_a < seed_b):
        seed_auc, auroc_auc = seed_a, auroc_a
        auroc_inst = auroc_b
    else:
        seed_auc, auroc_auc = seed_b, auroc_b
        auroc_inst = auroc_a

    if inst_a < inst_b or (np.isclose(inst_a, inst_b, rtol=0, atol=1e-12) and seed_a < seed_b):
        seed_inst = seed_a
    else:
        seed_inst = seed_b

    inst_auc = inst_a if seed_auc == seed_a else inst_b
    inst_inst = inst_a if seed_inst == seed_a else inst_b
    return seed_auc, seed_inst, auroc_auc, auroc_inst, inst_auc, inst_inst


def build_near_auroc_pair_cases(
    setting_df: pd.DataFrame,
    rc_lookup: Dict[Tuple[str, str, int, int], pd.DataFrame],
    *,
    epsilon: float = DEFAULT_EPSILON,
    tau: float = DEFAULT_TAU,
    coverages: Iterable[float] = DEFAULT_COVERAGES_FIG7,
    epsilons: Iterable[float] = DEFAULT_EPSILONS_FIG7,
) -> pd.DataFrame:
    """One row per (pair, coverage, epsilon) in the near-AUROC universe."""
    rows: List[Dict[str, Any]] = []
    cov_list = list(coverages)
    eps_list = list(epsilons)

    for (d, cat, shot), g in setting_df.groupby(["dataset", "category", "shot"], sort=False):
        recs = g.to_dict(orient="records")
        n = len(recs)
        if n < 2:
            continue
        for i in range(n):
            for j in range(i + 1, n):
                ra, rb = recs[i], recs[j]
                sa, sb = int(ra["seed"]), int(rb["seed"])
                aa = float(ra["setting_auroc"])
                ab = float(rb["setting_auroc"])
                if abs(aa - ab) >= float(epsilon):
                    continue
                ia = float(ra["mean_pairwise_instability"])
                ib = float(rb["mean_pairwise_instability"])
                seed_auc, seed_inst, auroc_auc, auroc_inst, inst_auc, inst_inst = assign_roles(
                    sa, sb, aa, ab, ia, ib
                )
                for eps in eps_list:
                    for cov in cov_list:
                        rc_a = rc_lookup.get((str(d), str(cat), int(shot), sa))
                        rc_b = rc_lookup.get((str(d), str(cat), int(shot), sb))
                        if rc_a is None or rc_b is None:
                            continue
                        risk_a = _risk_at_coverage(rc_a, cov)
                        risk_b = _risk_at_coverage(rc_b, cov)
                        if not (np.isfinite(risk_a) and np.isfinite(risk_b)):
                            continue
                        risk_auc = risk_a if seed_auc == sa else risk_b
                        risk_inst = risk_a if seed_inst == sa else risk_b
                        risks = {sa: risk_a, sb: risk_b}
                        seed_oracle = int(min(risks, key=lambda k: risks[k]))
                        risk_oracle = risks[seed_oracle]
                        delta_risk = float(risk_auc - risk_inst)
                        rows.append(
                            {
                                "dataset": d,
                                "category": cat,
                                "shot": int(shot),
                                "seed_a": sa,
                                "seed_b": sb,
                                "delta_auroc_pair": float(abs(aa - ab)),
                                "epsilon": float(eps),
                                "coverage": float(cov),
                                "seed_auc": int(seed_auc),
                                "seed_inst": int(seed_inst),
                                "seed_oracle": seed_oracle,
                                "auroc_auc": float(auroc_auc),
                                "auroc_inst": float(auroc_inst),
                                "instability_auc_seed": float(inst_auc),
                                "instability_inst_seed": float(inst_inst),
                                "risk_auc": float(risk_auc),
                                "risk_inst": float(risk_inst),
                                "risk_oracle": float(risk_oracle),
                                "delta_risk": delta_risk,
                                "failure": bool(delta_risk > float(tau)),
                                "gap_auc": float(risk_auc - risk_oracle),
                            }
                        )
    return pd.DataFrame(rows)


def aggregate_fig7(cases: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (eps, cov), g in cases.groupby(["epsilon", "coverage"], sort=True):
        dr = g["delta_risk"].astype(float).to_numpy()
        rows.append(
            {
                "epsilon": float(eps),
                "coverage": float(cov),
                "n_pairs": int(len(g)),
                "mean_delta_risk": float(np.mean(dr)),
                "median_delta_risk": float(np.median(dr)),
                "win_rate_instability_better": float(np.mean(dr > 0)),
                "failure_rate": float(np.mean(g["failure"])),
            }
        )
    return pd.DataFrame(rows).sort_values(["epsilon", "coverage"], kind="mergesort")


def failure_gate_from_cases(
    cases: pd.DataFrame,
    *,
    epsilon: float = DEFAULT_EPSILON,
    coverages: Iterable[float] = DEFAULT_COVERAGES_FIG56_REGIME,
) -> pd.DataFrame:
    sub = cases[np.isclose(cases["epsilon"].astype(float), float(epsilon), rtol=0, atol=1e-12)]
    sub = sub[sub["coverage"].isin(list(coverages))]
    if sub.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for (cov,), g_cov in sub.groupby(["coverage"], sort=True):
        # Regime driver: max instability across the two near-AUROC seeds (pair ambiguity).
        inst = np.maximum(
            g_cov["instability_auc_seed"].astype(float).to_numpy(),
            g_cov["instability_inst_seed"].astype(float).to_numpy(),
        )
        if len(inst) < 3:
            q30, q70 = np.quantile(inst, [0.33, 0.66])
        else:
            q30, q70 = np.quantile(inst, [0.3, 0.7])
        regime = np.where(inst <= q30, "low_I", np.where(inst >= q70, "high_I", "mid_I"))
        g_cov = g_cov.copy()
        g_cov["regime"] = regime
        for reg, gr in g_cov.groupby("regime", sort=False):
            rows.append(
                {
                    "epsilon": float(epsilon),
                    "coverage": float(cov),
                    "regime": reg,
                    "n_settings": int(len(gr)),
                    "failure_rate": float(gr["failure"].mean()),
                    "mean_delta_risk": float(gr["delta_risk"].mean()),
                }
            )
    return pd.DataFrame(rows)


def weighted_regime_failure_rates(gate: pd.DataFrame) -> np.ndarray:
    out = np.full(3, np.nan, dtype=np.float64)
    for i, reg in enumerate(REGIME_ORDER):
        sub = gate[gate["regime"].astype(str) == reg]
        if sub.empty:
            continue
        w = sub["n_settings"].astype(float).to_numpy()
        fr = sub["failure_rate"].astype(float).to_numpy()
        tot = float(np.sum(w))
        out[i] = float(np.sum(fr * w) / tot) if tot > 0 else np.nan
    return out


def failure_conditioned_signals_from_cases(
    cases: pd.DataFrame,
    signal_df: pd.DataFrame,
    *,
    epsilon: float = DEFAULT_EPSILON,
    coverages: Iterable[float] = DEFAULT_COVERAGES_FIG56_REGIME,
) -> pd.DataFrame:
    """
    signal_df: one row per (dataset, category, shot, seed) with columns
    instability, gap, score_var, margin (setting-level).
    """
    sub = cases[np.isclose(cases["epsilon"].astype(float), float(epsilon), rtol=0, atol=1e-12)]
    sub = sub[sub["coverage"].isin(list(coverages))]
    if sub.empty:
        return pd.DataFrame()

    merged_rows: List[Dict[str, Any]] = []
    for _, r in sub.iterrows():
        d, cat, shot = str(r["dataset"]), str(r["category"]), int(r["shot"])
        sa = int(r["seed_auc"])
        key = (d, cat, shot, sa)
        sig = signal_df[
            (signal_df["dataset"] == d)
            & (signal_df["category"] == cat)
            & (signal_df["shot"] == shot)
            & (signal_df["seed"] == sa)
        ]
        if sig.empty:
            continue
        row = sig.iloc[0].to_dict()
        row["failure"] = bool(r["failure"])
        row["coverage"] = float(r["coverage"])
        merged_rows.append(row)
    if not merged_rows:
        return pd.DataFrame()
    m = pd.DataFrame(merged_rows)
    rows: List[Dict[str, Any]] = []
    for sig in SIGNAL_ORDER:
        if sig not in m.columns:
            continue
        gf = m[m["failure"] == True][sig].astype(float)  # noqa: E712
        gn = m[m["failure"] == False][sig].astype(float)  # noqa: E712
        rows.append(
            {
                "signal": sig,
                "mean_failure": float(gf.mean()) if len(gf) else float("nan"),
                "mean_non_failure": float(gn.mean()) if len(gn) else float("nan"),
                "n_failure": int(len(gf)),
                "n_non_failure": int(len(gn)),
            }
        )
    return pd.DataFrame(rows)


def compute_setting_signals_from_wide(
    wide_path: Path,
    keys: Set[Tuple[str, str, int, int]],
    *,
    chunksize: int = 200_000,
) -> pd.DataFrame:
    """Setting-level gap / score_var / margin from cached wide raw (no Stage-1 rerun)."""
    usecols = [
        "dataset",
        "category",
        "shot",
        "seed",
        "label",
        "fused_score",
        "semantic_score",
        "visual_score",
    ]
    acc: Dict[Tuple[str, str, int, int], Dict[str, List[float]]] = {}
    for chunk in pd.read_csv(wide_path, usecols=usecols, chunksize=chunksize):
        chunk = chunk.dropna(subset=["fused_score", "semantic_score", "visual_score"])
        for (d, cat, shot, sd), g in chunk.groupby(["dataset", "category", "shot", "seed"], sort=False):
            key = (str(d), str(cat), int(shot), int(sd))
            if key not in keys:
                continue
            sem = g["semantic_score"].astype(float).to_numpy()
            vis = g["visual_score"].astype(float).to_numpy()
            fused = g["fused_score"].astype(float).to_numpy()
            lab = g["label"].astype(int).to_numpy()
            if key not in acc:
                acc[key] = {"gap": [], "fused": [], "anom_fused": [], "norm_fused": []}
            acc[key]["gap"].extend(np.abs(sem - vis).tolist())
            acc[key]["fused"].extend(fused.tolist())
            acc[key]["anom_fused"].extend(fused[lab == 1].tolist())
            acc[key]["norm_fused"].extend(fused[lab == 0].tolist())

    rows: List[Dict[str, Any]] = []
    for key in keys:
        d, cat, shot, sd = key
        if key not in acc:
            continue
        a = acc[key]
        gap_m = float(np.mean(a["gap"])) if a["gap"] else float("nan")
        score_var = float(np.var(a["fused"])) if len(a["fused"]) > 1 else float("nan")
        anom = np.array(a["anom_fused"], dtype=float)
        norm = np.array(a["norm_fused"], dtype=float)
        if anom.size and norm.size:
            margin = float(np.mean(np.abs(anom[:, None] - norm[None, :])))
        else:
            margin = float("nan")
        rows.append(
            {
                "dataset": d,
                "category": cat,
                "shot": shot,
                "seed": sd,
                "gap": gap_m,
                "score_var": score_var,
                "margin": margin,
            }
        )
    return pd.DataFrame(rows)


def build_setting_signals(
    setting_df: pd.DataFrame,
    cases: pd.DataFrame,
    wide_path: Optional[Path],
) -> Tuple[pd.DataFrame, List[str]]:
    """Join setting instability + optional wide-derived gap/score_var/margin."""
    seeds_needed: Set[Tuple[str, str, int, int]] = set()
    for _, r in cases.iterrows():
        d, cat, shot = str(r["dataset"]), str(r["category"]), int(r["shot"])
        seeds_needed.add((d, cat, shot, int(r["seed_auc"])))

    base = setting_df[
        ["dataset", "category", "shot", "seed", "mean_pairwise_instability"]
    ].rename(columns={"mean_pairwise_instability": "instability"})
    warnings: List[str] = []
    if wide_path is not None and wide_path.is_file():
        extra = compute_setting_signals_from_wide(wide_path, seeds_needed)
        out = base.merge(extra, on=["dataset", "category", "shot", "seed"], how="left")
    else:
        out = base.copy()
        out["gap"] = np.nan
        out["score_var"] = np.nan
        out["margin"] = np.nan
        warnings.append("wide raw missing: gap/score_var/margin unavailable (instability only)")
    return out, warnings


_PAIRWISE_FIG34_COLS = [
    "dataset",
    "category",
    "shot",
    "seed",
    "abs_fused_margin",
    "pairwise_instability",
    "pairwise_error",
]


def load_pairwise_for_pair_universe(
    pairwise_path: Path,
    cases: pd.DataFrame,
    *,
    epsilon: float = DEFAULT_EPSILON,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Load only rows whose (dataset, category, shot, seed) appear in the near-AUROC universe."""
    sub_cases = cases[np.isclose(cases["epsilon"].astype(float), float(epsilon), rtol=0, atol=1e-12)]
    if sub_cases.empty:
        return pd.DataFrame()
    keys_auc = sub_cases[["dataset", "category", "shot", "seed_auc"]].rename(columns={"seed_auc": "seed"})
    keys_inst = sub_cases[["dataset", "category", "shot", "seed_inst"]].rename(columns={"seed_inst": "seed"})
    keep = pd.concat([keys_auc, keys_inst], ignore_index=True).drop_duplicates()
    parts: List[pd.DataFrame] = []
    for chunk in pd.read_csv(pairwise_path, usecols=_PAIRWISE_FIG34_COLS, chunksize=int(chunksize)):
        m = chunk.merge(keep, on=["dataset", "category", "shot", "seed"], how="inner")
        if len(m):
            parts.append(m)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def conditioned_pairwise_margin_tables(
    pairwise: pd.DataFrame,
    cases: pd.DataFrame,
    *,
    epsilon: float = DEFAULT_EPSILON,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Strict pairwise-level conditioning for fig3_4 (a)(b).

    Restrict ``pairwise_metrics.csv`` to (dataset, category, shot, seed) tuples that
    appear as either role in the near-AUROC pair universe at the given epsilon, then
    compute margin tertiles and low-/high-instability tails **on pair rows directly**
    (no per-setting averaging proxy).
    """
    sub_cases = cases[np.isclose(cases["epsilon"].astype(float), float(epsilon), rtol=0, atol=1e-12)]
    if sub_cases.empty or pairwise.empty:
        return pd.DataFrame(), pd.DataFrame()

    pw = pairwise
    abs_m = pw["abs_fused_margin"].to_numpy(dtype=float)
    if len(abs_m) < 6:
        return pd.DataFrame(), pd.DataFrame()

    q1, q2 = np.quantile(abs_m, [1.0 / 3.0, 2.0 / 3.0])
    bucket_arr = np.where(abs_m <= q1, "low", np.where(abs_m <= q2, "mid", "high"))
    pw = pw.assign(margin_bucket=bucket_arr)

    chain_rows: List[Dict[str, Any]] = []
    cm_rows: List[Dict[str, Any]] = []
    for b in ["low", "mid", "high"]:
        sub = pw[pw["margin_bucket"] == b]
        if len(sub) < 6:
            continue
        inst = sub["pairwise_instability"].to_numpy(dtype=float)
        err = sub["pairwise_error"].to_numpy(dtype=float)
        abs_sub = sub["abs_fused_margin"].to_numpy(dtype=float)
        n_sub = len(inst)
        iq1, iq2 = np.quantile(inst, [1.0 / 3.0, 2.0 / 3.0])
        m0 = inst <= iq1
        m1 = inst >= iq2
        if m0.sum() == 0 or m1.sum() == 0:
            k = max(1, n_sub // 3)
            ridx = np.argsort(inst, kind="mergesort")
            m0 = np.zeros(n_sub, dtype=bool)
            m1 = np.zeros(n_sub, dtype=bool)
            m0[ridx[:k]] = True
            m1[ridx[-k:]] = True
        if m0.sum() == 0 or m1.sum() == 0:
            continue
        chain_rows.append(
            {
                "margin_bucket": b,
                "mean_instability": float(np.mean(inst)),
                "error_rate": float(np.mean(err)),
                "mean_abs_margin": float(np.mean(abs_sub)),
                "n_pairs": int(n_sub),
            }
        )
        cm_rows.append(
            {
                "margin_bucket": b,
                "n_low_I": int(m0.sum()),
                "n_high_I": int(m1.sum()),
                "error_low_I": float(np.mean(err[m0])),
                "error_high_I": float(np.mean(err[m1])),
                "mean_margin_low_I": float(np.mean(abs_sub[m0])),
                "mean_margin_high_I": float(np.mean(abs_sub[m1])),
            }
        )
    return pd.DataFrame(chain_rows), pd.DataFrame(cm_rows)


def conditioned_margin_tables(
    cm_detail: pd.DataFrame,
    cases: pd.DataFrame,
    *,
    epsilon: float = DEFAULT_EPSILON,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict mechanism panels (a)(b) to settings that appear in the pair universe."""
    settings = cases[np.isclose(cases["epsilon"].astype(float), float(epsilon), rtol=0, atol=1e-12)][
        ["dataset", "category", "shot"]
    ].drop_duplicates()
    d = cm_detail.copy()
    if "scope" in d.columns:
        d = d[d["scope"].astype(str) == "per_setting"]
    m = d.merge(settings, on=["dataset", "category", "shot"], how="inner")
    chain_rows = []
    cm_rows = []
    for b in ["low", "mid", "high"]:
        sub = m[m["margin_bucket"].astype(str).str.lower() == b]
        if sub.empty:
            continue
        w = sub["n_low_I"].astype(float) + sub["n_high_I"].astype(float)
        e0 = float(np.average(sub["error_low_I"], weights=w)) if w.sum() > 0 else float("nan")
        e1 = float(np.average(sub["error_high_I"], weights=w)) if w.sum() > 0 else float("nan")
        chain_rows.append(
            {
                "margin_bucket": b,
                "mean_instability": float(np.average(sub["mean_instability"], weights=w)),
                "error_rate": float(np.average(sub["error_rate"], weights=w)),
            }
        )
        n0 = int(sub["n_low_I"].sum())
        n1 = int(sub["n_high_I"].sum())
        cm_rows.append(
            {
                "margin_bucket": b,
                "n_low_I": n0,
                "n_high_I": n1,
                "error_low_I": e0,
                "error_high_I": e1,
            }
        )
    return pd.DataFrame(chain_rows), pd.DataFrame(cm_rows)


def pick_fig2_setting(
    cases: pd.DataFrame,
    *,
    epsilon: float = DEFAULT_EPSILON,
    tau: float = DEFAULT_TAU,
) -> pd.Series:
    """
    Choose an exemplar (dataset, category, shot, coverage) from the near-AUROC pair universe.

    Ranking uses pair-level failure evidence (Δrisk, instability separation) only to
    *select the setting*; role markers on the figure are assigned separately at setting scope.
    """
    sub = cases[np.isclose(cases["epsilon"].astype(float), float(epsilon), rtol=0, atol=1e-12)]
    if sub.empty:
        raise ValueError(f"no near-AUROC pair rows at epsilon={epsilon}")
    clean = sub[
        (sub["delta_risk"] > float(tau))
        & (sub["instability_auc_seed"] > sub["instability_inst_seed"])
    ].copy()
    if clean.empty:
        clean = sub[sub["delta_risk"] > 0].copy()
    if clean.empty:
        raise ValueError("no fig2-eligible failure cases in near-AUROC universe")
    clean["instability_gap"] = clean["instability_auc_seed"] - clean["instability_inst_seed"]
    best = clean.sort_values(["delta_risk", "instability_gap"], ascending=False).iloc[0]
    return pd.Series(
        {
            "dataset": best["dataset"],
            "category": best["category"],
            "shot": int(best["shot"]),
            "coverage": float(best["coverage"]),
            "epsilon": float(epsilon),
            "exemplar_delta_risk_pair": float(best["delta_risk"]),
            "exemplar_instability_gap_pair": float(best["instability_gap"]),
            "exemplar_seed_a": int(best["seed_a"]),
            "exemplar_seed_b": int(best["seed_b"]),
        }
    )


def assign_setting_global_roles(
    setting_df: pd.DataFrame,
    rc_lookup: Dict[Tuple[str, str, int, int], pd.DataFrame],
    *,
    dataset: str,
    category: str,
    shot: int,
    coverage: float,
) -> Dict[str, Any]:
    """
    Setting-global roles for fig2 (tie-break: smaller seed id).

    - AUROC-selected: max setting_auroc
    - instability-selected: min mean_pairwise_instability
    - oracle: min decision risk @ coverage
    """
    g = setting_df[
        (setting_df["dataset"] == dataset)
        & (setting_df["category"] == category)
        & (setting_df["shot"] == shot)
    ].copy()
    if g.empty:
        raise ValueError(f"no seeds for setting {dataset}/{category}/k={shot}")

    rows: List[Dict[str, Any]] = []
    for _, r in g.iterrows():
        sd = int(r["seed"])
        rc = rc_lookup.get((str(dataset), str(category), int(shot), sd))
        if rc is None:
            continue
        risk = _risk_at_coverage(rc, float(coverage))
        if not np.isfinite(risk):
            continue
        rows.append(
            {
                "seed": sd,
                "setting_auroc": float(r["setting_auroc"]),
                "mean_pairwise_instability": float(r["mean_pairwise_instability"]),
                "risk": float(risk),
            }
        )
    if not rows:
        raise ValueError(f"no finite risks at coverage={coverage} for {dataset}/{category}/k={shot}")
    tab = pd.DataFrame(rows)

    seed_auc = int(
        tab.sort_values(["setting_auroc", "seed"], ascending=[False, True]).iloc[0]["seed"]
    )
    seed_inst = int(
        tab.sort_values(["mean_pairwise_instability", "seed"], ascending=[True, True]).iloc[0]["seed"]
    )
    seed_oracle = int(tab.sort_values(["risk", "seed"], ascending=[True, True]).iloc[0]["seed"])

    row_auc = tab[tab["seed"] == seed_auc].iloc[0]
    row_inst = tab[tab["seed"] == seed_inst].iloc[0]
    row_orc = tab[tab["seed"] == seed_oracle].iloc[0]

    return {
        "seed_auc": seed_auc,
        "seed_inst": seed_inst,
        "seed_oracle": seed_oracle,
        "auroc_selected": float(row_auc["setting_auroc"]),
        "auroc_instability_selected": float(row_inst["setting_auroc"]),
        "instability_selected": float(row_inst["mean_pairwise_instability"]),
        "instability_auroc_selected": float(row_auc["mean_pairwise_instability"]),
        "risk_auc": float(row_auc["risk"]),
        "risk_inst": float(row_inst["risk"]),
        "risk_oracle": float(row_orc["risk"]),
        "delta_risk_global_roles": float(row_auc["risk"] - row_inst["risk"]),
        "n_seeds_plotted": int(len(tab)),
    }
