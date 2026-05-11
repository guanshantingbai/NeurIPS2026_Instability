#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from pilot_instability_aware_selection import (
    EPS,
    FINAL_SCORE_CANDIDATES,
    LABEL_CANDIDATES,
    PATH_CANDIDATES,
    SEM_SCORE_CANDIDATES,
    VIS_SCORE_CANDIDATES,
    _discover_per_sample_csvs,
    _first_existing_column,
    _parse_setting_seed_from_path,
    _to_label01,
)


EPSILONS = (0.002, 0.005)
COVERAGES = (0.9, 0.8, 0.7, 0.5, 0.3)


def _savefig_both(fig: plt.Figure, out_no_ext: str) -> None:
    fig.savefig(out_no_ext + ".png", dpi=180, bbox_inches="tight")
    fig.savefig(out_no_ext + ".pdf", bbox_inches="tight")
    plt.close(fig)


class WarnLog:
    def __init__(self) -> None:
        self.rows: List[Dict[str, str]] = []

    def add(self, scope: str, msg: str, detail: str = "") -> None:
        self.rows.append({"scope": scope, "message": msg, "detail": detail})


def _load_std_per_sample(df: pd.DataFrame) -> pd.DataFrame:
    label_col = _first_existing_column(df, LABEL_CANDIDATES)
    sem_col = _first_existing_column(df, SEM_SCORE_CANDIDATES)
    vis_col = _first_existing_column(df, VIS_SCORE_CANDIDATES)
    final_col = _first_existing_column(df, FINAL_SCORE_CANDIDATES)
    path_col = _first_existing_column(df, PATH_CANDIDATES)
    if label_col is None or sem_col is None or vis_col is None:
        raise ValueError("missing label/semantic/visual")
    out = pd.DataFrame()
    out["image_path"] = df[path_col].astype(str) if path_col is not None else np.arange(len(df)).astype(str)
    out["image_label"] = _to_label01(df[label_col])
    out["semantic_score"] = pd.to_numeric(df[sem_col], errors="coerce")
    out["visual_score"] = pd.to_numeric(df[vis_col], errors="coerce")
    if final_col is not None:
        out["final_score"] = pd.to_numeric(df[final_col], errors="coerce")
    else:
        out["final_score"] = np.nan
    miss = out["final_score"].isna()
    if miss.any():
        a = out.loc[miss, "semantic_score"].to_numpy(dtype=np.float64)
        b = out.loc[miss, "visual_score"].to_numpy(dtype=np.float64)
        out.loc[miss, "final_score"] = (2.0 * a * b) / (a + b + EPS)
    out = out.dropna(subset=["image_label", "semantic_score", "visual_score", "final_score"]).copy()
    out["image_label"] = out["image_label"].astype(int)
    out = out[(out["image_label"] == 0) | (out["image_label"] == 1)]
    return out


def _bootstrap_setting_ci(df: pd.DataFrame, value_col: str, n_boot: int = 1000, seed: int = 42) -> Tuple[float, float]:
    if df.empty:
        return float("nan"), float("nan")
    keys = df[["dataset", "class", "k"]].drop_duplicates().to_records(index=False).tolist()
    keys = [(str(a), str(b), int(c)) for a, b, c in keys]
    if len(keys) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick_idx = rng.choice(len(keys), size=len(keys), replace=True)
        parts = []
        for i in pick_idx:
            d, c, k = keys[i]
            parts.append(df[(df["dataset"] == d) & (df["class"] == c) & (df["k"] == k)])
        sub = pd.concat(parts, axis=0)
        vals.append(float(np.nanmean(pd.to_numeric(sub[value_col], errors="coerce"))))
    arr = np.array([v for v in vals if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))


def run(args: argparse.Namespace) -> None:
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    wlog = WarnLog()

    pilot_dir = os.path.abspath(args.pilot_dir)
    strengthening_dir = os.path.abspath(args.strengthening_dir)
    failure_path = os.path.join(pilot_dir, "failure_driven", "failure_analysis.csv")
    failure_df = pd.read_csv(failure_path)
    # S2 failure definition for this analysis:
    failure_df["failure_s2"] = pd.to_numeric(failure_df["delta_risk"], errors="coerce") > 0.01

    # Recover seed_auc via selection summary (failure_analysis may not contain it)
    sel_list = []
    for eps in EPSILONS:
        p = os.path.join(pilot_dir, f"selection_summary_epsilon_{eps:.3f}.csv")
        if os.path.isfile(p):
            sel_list.append(pd.read_csv(p))
    if not sel_list:
        raise SystemExit("selection_summary files not found")
    sel = pd.concat(sel_list, axis=0, ignore_index=True)
    key_cols = ["dataset", "class", "k", "coverage", "epsilon", "seed_auc", "auroc_auc_seed"]
    sel = sel[[c for c in key_cols if c in sel.columns]].drop_duplicates()
    merged = failure_df.merge(
        sel,
        on=[c for c in ["dataset", "class", "k", "coverage", "epsilon", "auroc_auc_seed"] if c in failure_df.columns and c in sel.columns],
        how="left",
    )
    if "seed_auc" not in merged.columns:
        merged["seed_auc"] = np.nan

    # Seed-level metrics
    per_seed = pd.read_csv(os.path.join(pilot_dir, "per_seed_metrics.csv"))
    inst_col = "setting_instability_flip" if "setting_instability_flip" in per_seed.columns else "setting_instability"
    per_seed = per_seed.rename(columns={inst_col: "setting_instability"})

    # Build seed signals from per-image raw scores
    seed_signals: Dict[Tuple[str, str, int, int], Dict[str, float]] = {}
    pair_rows: List[Dict[str, Any]] = []
    seen = set()
    for p in _discover_per_sample_csvs(args.promptad_root):
        parsed = _parse_setting_seed_from_path(p)
        if parsed is None:
            continue
        key = (parsed.dataset, parsed.cls, parsed.k, parsed.seed)
        if key in seen:
            continue
        try:
            df = pd.read_csv(p)
            w = _load_std_per_sample(df)
        except Exception as e:  # noqa: BLE001
            wlog.add("load", str(e), p)
            continue
        y = w["image_label"].to_numpy(dtype=np.int8)
        pos = np.where(y == 1)[0]
        neg = np.where(y == 0)[0]
        if len(pos) == 0 or len(neg) == 0:
            wlog.add("pair", "missing anomaly or normal", p)
            continue
        sem = w["semantic_score"].to_numpy(dtype=np.float64)
        vis = w["visual_score"].to_numpy(dtype=np.float64)
        fin = w["final_score"].to_numpy(dtype=np.float64)
        sp, sn = sem[pos], sem[neg]
        vp, vn = vis[pos], vis[neg]
        fp, fn = fin[pos], fin[neg]
        z_sem = sp[:, None] > sn[None, :]
        z_vis = vp[:, None] > vn[None, :]
        i_flip = (z_sem != z_vis).astype(np.float64)
        err = (fp[:, None] <= fn[None, :]).astype(np.float64)
        margin = (fp[:, None] - fn[None, :]).astype(np.float64)

        m2 = 0.5 * (sem + vis)
        score_var = 0.5 * ((sem - m2) ** 2 + (vis - m2) ** 2)
        gap = np.abs(sem - vis)
        # sample margin mean abs to opposite label
        mean_abs_margin_s = np.zeros(len(w), dtype=np.float64)
        mean_abs_margin_s[pos] = np.mean(np.abs(margin), axis=1)
        mean_abs_margin_s[neg] = np.mean(np.abs(margin), axis=0)

        seed_signals[key] = {
            "mean_instability": float(np.mean(i_flip)),
            "mean_margin": float(np.mean(np.abs(margin))),
            "mean_gap": float(np.mean(gap)),
            "mean_score_var": float(np.mean(score_var)),
        }

        # controlled margin source pairs
        abs_m = np.abs(margin).ravel()
        err_f = err.ravel()
        i_f = i_flip.ravel()
        if abs_m.size >= 3:
            q1, q2 = np.quantile(abs_m, [1 / 3, 2 / 3])
            mb = np.where(abs_m > q2, "high", np.where(abs_m > q1, "mid", "low"))
            for b in ("low", "mid", "high"):
                msk_b = mb == b
                if not np.any(msk_b):
                    continue
                m0 = msk_b & (i_f < 0.5)
                m1 = msk_b & (i_f > 0.5)
                if not np.any(m0) or not np.any(m1):
                    wlog.add("S3", "empty I group in margin bucket", f"{key}-{b}")
                    continue
                pair_rows.append(
                    {
                        "dataset": key[0],
                        "class": key[1],
                        "k": key[2],
                        "seed": key[3],
                        "margin_bucket": b,
                        "n_low_I": int(np.sum(m0)),
                        "n_high_I": int(np.sum(m1)),
                        "error_low_I": float(np.mean(err_f[m0])),
                        "error_high_I": float(np.mean(err_f[m1])),
                        "mean_margin_low_I": float(np.mean(abs_m[m0])),
                        "mean_margin_high_I": float(np.mean(abs_m[m1])),
                        "mean_instability": float(np.mean(i_f[msk_b])),
                        "error_rate": float(np.mean(err_f[msk_b])),
                    }
                )
        seen.add(key)

    # ---- S2 Failure Gate ----
    # add regime from instability_auc_seed (AUROC seed instability)
    fg = merged.copy()
    out_s2_rows = []
    for (eps, cov), g in fg.groupby(["epsilon", "coverage"], sort=True):
        inst = pd.to_numeric(g["instability_auc_seed"], errors="coerce")
        nz = inst.dropna().to_numpy(dtype=np.float64)
        if len(nz) == 0:
            continue
        q30, q70 = np.quantile(nz, [0.3, 0.7])
        regime = np.where(inst <= q30, "low_I", np.where(inst >= q70, "high_I", "mid_I"))
        g = g.copy()
        g["regime"] = regime
        for r, gr in g.groupby("regime", sort=False):
            out_s2_rows.append(
                {
                    "epsilon": float(eps),
                    "coverage": float(cov),
                    "regime": r,
                    "n_settings": int(len(gr)),
                    "failure_rate": float(np.mean(gr["failure_s2"])),
                    "mean_delta_risk": float(np.mean(gr["delta_risk"])),
                    "median_delta_risk": float(np.median(gr["delta_risk"])),
                    "mean_oracle_gap_auc": float(np.mean(gr["gap_auc"])),
                    "mean_oracle_gap_stable": float(np.mean(gr["gap_stable"])),
                    "mean_gap_reduction": float(np.mean(gr["gap_reduction"])),
                }
            )
    s2 = pd.DataFrame(out_s2_rows).sort_values(["epsilon", "coverage", "regime"])
    s2.to_csv(os.path.join(out_dir, "failure_gate_analysis.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    s2p = s2[s2["epsilon"] == 0.002]
    for cov in sorted(s2p["coverage"].unique()):
        sub = s2p[s2p["coverage"] == cov].set_index("regime").reindex(["low_I", "mid_I", "high_I"])
        ax.plot(["low_I", "mid_I", "high_I"], sub["failure_rate"], marker="o", label=f"cov={cov}")
    ax.set_ylabel("failure_rate")
    ax.set_xlabel("regime")
    ax.legend()
    ax.set_title("Failure gate by instability regime (eps=0.002)")
    _savefig_both(fig, os.path.join(out_dir, "failure_gate_bar"))

    # ---- S3 Controlled Margin ----
    pdf = pd.DataFrame(pair_rows)
    cm_rows = []
    for b, g in pdf.groupby("margin_bucket", sort=False):
        n0 = int(np.sum(g["n_low_I"]))
        n1 = int(np.sum(g["n_high_I"]))
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
                "error_gap_high_minus_low": e1 - e0 if np.isfinite(e0) and np.isfinite(e1) else float("nan"),
                "mean_margin_low_I": m0,
                "mean_margin_high_I": m1,
            }
        )
    cm = pd.DataFrame(cm_rows).sort_values("margin_bucket")
    cm.to_csv(os.path.join(out_dir, "controlled_margin_analysis.csv"), index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    order = ["low", "mid", "high"]
    x = np.arange(len(order))
    w = 0.35
    e0 = [cm.loc[cm["margin_bucket"] == b, "error_low_I"].iloc[0] if len(cm[cm["margin_bucket"] == b]) else np.nan for b in order]
    e1 = [cm.loc[cm["margin_bucket"] == b, "error_high_I"].iloc[0] if len(cm[cm["margin_bucket"] == b]) else np.nan for b in order]
    ax.bar(x - w / 2, e0, width=w, label="I=0")
    ax.bar(x + w / 2, e1, width=w, label="I=1")
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_xlabel("margin_bucket")
    ax.set_ylabel("error")
    ax.legend()
    ax.set_title("Controlled margin analysis")
    _savefig_both(fig, os.path.join(out_dir, "controlled_margin_bar"))

    # ---- S1 Fragile Correctness ----
    # group based on failure_s2 over setting+eps+cov rows
    s1_rows = []
    x = merged.copy()
    def _attach_seed_signals(row: pd.Series) -> pd.Series:
        key = (str(row["dataset"]), str(row["class"]), int(row["k"]), int(row["seed_auc"])) if pd.notna(row["seed_auc"]) else None
        if key is None or key not in seed_signals:
            return pd.Series({"mean_instability": np.nan, "mean_margin": np.nan, "mean_gap": np.nan, "mean_score_var": np.nan})
        return pd.Series(seed_signals[key])

    x[["mean_instability", "mean_margin", "mean_gap", "mean_score_var"]] = x.apply(_attach_seed_signals, axis=1)
    for grp_name, gx in [("failure", x[x["failure_s2"]]), ("non_failure", x[~x["failure_s2"]])]:
        s1_rows.append(
            {
                "group": grp_name,
                "mean_instability": float(np.nanmean(gx["mean_instability"])),
                "median_instability": float(np.nanmedian(gx["mean_instability"])),
                "mean_margin": float(np.nanmean(gx["mean_margin"])),
                "median_margin": float(np.nanmedian(gx["mean_margin"])),
                "mean_gap": float(np.nanmean(gx["mean_gap"])),
                "median_gap": float(np.nanmedian(gx["mean_gap"])),
                "mean_score_var": float(np.nanmean(gx["mean_score_var"])),
                "median_score_var": float(np.nanmedian(gx["mean_score_var"])),
            }
        )
    s1 = pd.DataFrame(s1_rows)
    s1.to_csv(os.path.join(out_dir, "fragile_correctness_summary.csv"), index=False)

    # ---- A1 Failure-conditioned separation (compact) ----
    fc_path = os.path.join(strengthening_dir, "failure_conditioned_signal_analysis.csv")
    fc = pd.read_csv(fc_path)
    fs = fc[
        fc["coverage"].isin([0.5, 0.7, 0.8]) & fc["signal"].isin(["instability", "gap", "score_var", "margin"])
    ][["coverage", "signal", "mean_failure", "mean_non_failure", "ratio"]].copy()
    fs = fs.sort_values(["coverage", "signal"]).reset_index(drop=True)
    fs.to_csv(os.path.join(out_dir, "failure_signal_summary.csv"), index=False)

    # ---- A2 Mechanism chain summary ----
    # failure rate mapped by setting (mean over eps/cov for this new failure_s2 def)
    fr_setting = (
        merged.groupby(["dataset", "class", "k"], as_index=False)["failure_s2"].mean().rename(columns={"failure_s2": "failure_rate_setting"})
    )
    pdf2 = pdf.merge(fr_setting, on=["dataset", "class", "k"], how="left")
    mc_rows = []
    for b, g in pdf2.groupby("margin_bucket", sort=False):
        mc_rows.append(
            {
                "margin_bucket": b,
                "mean_instability": float(np.mean(g["mean_instability"])),
                "error_rate": float(np.mean(g["error_rate"])),
                "failure_rate": float(np.mean(g["failure_rate_setting"])),
            }
        )
    mc = pd.DataFrame(mc_rows).sort_values("margin_bucket")
    mc.to_csv(os.path.join(out_dir, "mechanism_chain_summary.csv"), index=False)

    # ---- B1 simple mediation/logit ----
    med_df = x[["failure_s2", "mean_margin", "mean_instability", "mean_gap", "mean_score_var"]].dropna().copy()
    med_rows = []
    if not med_df.empty and med_df["failure_s2"].nunique() >= 2:
        y = med_df["failure_s2"].astype(int).to_numpy()
        specs = [
            ("margin", ["mean_margin"]),
            ("margin+instability", ["mean_margin", "mean_instability"]),
            ("margin+instability+gap+score_var", ["mean_margin", "mean_instability", "mean_gap", "mean_score_var"]),
        ]
        null_p = float(np.mean(y))
        null_prob = np.full_like(y, fill_value=null_p, dtype=np.float64)
        ll_null = log_loss(y, np.clip(null_prob, 1e-6, 1 - 1e-6), normalize=True)
        for name, cols in specs:
            X = med_df[cols].to_numpy(dtype=np.float64)
            model = LogisticRegression(max_iter=2000)
            model.fit(X, y)
            prob = model.predict_proba(X)[:, 1]
            ll = log_loss(y, np.clip(prob, 1e-6, 1 - 1e-6), normalize=True)
            auc = roc_auc_score(y, prob)
            pseudo_r2 = 1.0 - (ll / ll_null) if ll_null > 0 else float("nan")
            med_rows.append({"model": name, "logloss": ll, "auc": auc, "pseudo_r2": pseudo_r2})
    else:
        wlog.add("B1", "insufficient data or single class for logistic regression")
    pd.DataFrame(med_rows).to_csv(os.path.join(out_dir, "mediation_analysis.csv"), index=False)

    # ---- bootstrap summary (setting-level resample) ----
    b_rows = []
    for (eps, cov), g in merged.groupby(["epsilon", "coverage"], sort=True):
        lo, hi = _bootstrap_setting_ci(g, "failure_s2")
        b_rows.append(
            {
                "epsilon": float(eps),
                "coverage": float(cov),
                "metric": "failure_rate",
                "mean": float(np.mean(g["failure_s2"])),
                "ci_low": lo,
                "ci_high": hi,
                "warning": "n_failure<10" if int(np.sum(g["failure_s2"])) < 10 else "",
            }
        )
        lo, hi = _bootstrap_setting_ci(g, "delta_risk")
        b_rows.append(
            {
                "epsilon": float(eps),
                "coverage": float(cov),
                "metric": "delta_risk_mean",
                "mean": float(np.mean(g["delta_risk"])),
                "ci_low": lo,
                "ci_high": hi,
                "warning": "",
            }
        )
    pd.DataFrame(b_rows).to_csv(os.path.join(out_dir, "bootstrap_setting_summary.csv"), index=False)

    # ---- README / claims ----
    c1 = bool((cm["error_gap_high_minus_low"].dropna() > 0).all()) if not cm.empty else False
    # compare instability vs gap/score_var on A1 ratios
    fs_m = fs.groupby("signal", as_index=False)["ratio"].mean() if not fs.empty else pd.DataFrame()
    inst_ratio = float(fs_m[fs_m["signal"] == "instability"]["ratio"].iloc[0]) if len(fs_m[fs_m["signal"] == "instability"]) else float("nan")
    gap_ratio = float(fs_m[fs_m["signal"] == "gap"]["ratio"].iloc[0]) if len(fs_m[fs_m["signal"] == "gap"]) else float("nan")
    var_ratio = float(fs_m[fs_m["signal"] == "score_var"]["ratio"].iloc[0]) if len(fs_m[fs_m["signal"] == "score_var"]) else float("nan")
    c2 = bool(s1.loc[s1["group"] == "failure", "mean_instability"].iloc[0] > s1.loc[s1["group"] == "non_failure", "mean_instability"].iloc[0]) if len(s1) == 2 else False
    # high-I failure concentration test from S2
    s2g = s2[s2["coverage"].isin([0.5, 0.7, 0.8])]
    c3 = False
    if not s2g.empty:
        high = s2g[s2g["regime"] == "high_I"]["failure_rate"].mean()
        low = s2g[s2g["regime"] == "low_I"]["failure_rate"].mean()
        c3 = bool(high > low)
    c4 = bool(np.isfinite(inst_ratio) and np.isfinite(gap_ratio) and np.isfinite(var_ratio) and inst_ratio > gap_ratio and inst_ratio > var_ratio)

    readme = [
        "# mechanism-driven analysis",
        "",
        "This directory contains S1/S2/S3/A1/A2/B1 analyses requested for PromptAD.",
        "",
        "## Key automatic claim checks",
        f"- Claim 1: instability != margin (controlled margin): **{c1}**",
        f"- Claim 2: instability identifies failure: **{c2}**",
        f"- Claim 3: failure concentrates in high-I regime: **{c3}**",
        f"- Claim 4: instability stronger than gap/variance: **{c4}**",
        "",
        "## Notes",
        "- All summary tables include mean and median where requested.",
        "- Bootstrap is setting-level (resample unit: setting), saved in `bootstrap_setting_summary.csv`.",
        "- coverage=0.3 should be treated as boundary case.",
    ]
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(readme))

    pd.DataFrame(wlog.rows).to_csv(os.path.join(out_dir, "warnings.csv"), index=False)
    print(f"Done. wrote={out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--promptad-root", default="/home/zju/mywork/NeurIPS2026/PromptAD")
    p.add_argument("--pilot-dir", default="/home/zju/mywork/NeurIPS2026/PromptAD/result_analysis/pilot_instability_selection")
    p.add_argument("--strengthening-dir", default="/home/zju/mywork/NeurIPS2026/PromptAD/result_analysis/promptad_strengthening")
    p.add_argument("--out-dir", default="/home/zju/mywork/NeurIPS2026/PromptAD/result_analysis/mechanism_analysis")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
