#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EPS = 1e-12
DEFAULT_COVERAGES = (0.9, 0.8, 0.7, 0.5, 0.3)
DEFAULT_EPSILONS = (0.002, 0.005)

FINAL_SCORE_CANDIDATES = ["harmonic_score", "final_score", "anomaly_score", "s_final", "score"]
SEM_SCORE_CANDIDATES = ["semantic_score", "s_sem", "sem_score"]
VIS_SCORE_CANDIDATES = ["visual_score", "s_vis", "vis_score"]
LABEL_CANDIDATES = ["image_label", "label", "y", "target", "gt", "is_anomaly"]
PATH_CANDIDATES = ["image_path", "path", "img_path"]


@dataclass(frozen=True)
class SettingSeed:
    dataset: str
    cls: str
    k: int
    seed: int


def _first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        m = lower.get(c.lower())
        if m is not None:
            return m
    return None


def _to_label01(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        x = pd.to_numeric(series, errors="coerce")
        return x.apply(lambda v: np.nan if pd.isna(v) else (1 if float(v) > 0.5 else 0))
    s = series.astype(str).str.strip().str.lower()
    pos = {"1", "anomaly", "abnormal", "positive", "pos", "true", "yes", "defect", "bad"}
    neg = {"0", "normal", "negative", "neg", "false", "no", "good"}
    return s.apply(lambda v: 1 if v in pos else (0 if v in neg else np.nan))


def _parse_setting_seed_from_path(path: str) -> Optional[SettingSeed]:
    norm = path.replace("\\", "/")
    m = re.search(r"/result_seed_search/([^/]+)/(\d+)/", norm)
    if m:
        slug = m.group(1)
        seed = int(m.group(2))
        m2 = re.match(r"(mvtec|visa)__(.+)__k(\d+)", slug)
        if m2:
            return SettingSeed(dataset=m2.group(1), cls=m2.group(2), k=int(m2.group(3)), seed=seed)

    base = os.path.basename(path)
    m3 = re.match(r"CLS-([^-]+)-(.+)-k(\d+)-seed(\d+)-per_sample\.csv", base)
    if m3:
        return SettingSeed(dataset=m3.group(1), cls=m3.group(2), k=int(m3.group(3)), seed=int(m3.group(4)))
    return None


def _discover_per_sample_csvs(promptad_root: str) -> List[str]:
    root = os.path.abspath(promptad_root)
    out: List[str] = []
    for base in [
        os.path.join(root, "result_seed_search"),
        os.path.join(root, "result_round1"),
        os.path.join(root, "result_analysis"),
    ]:
        if not os.path.isdir(base):
            continue
        for dirpath, _, filenames in os.walk(base):
            for fn in filenames:
                if fn.endswith("-per_sample.csv") and "per_sample_instability" not in fn:
                    out.append(os.path.join(dirpath, fn))
    return sorted(set(out))


def _auc_from_scores(scores: np.ndarray, labels01: np.ndarray) -> float:
    pos = scores[labels01 == 1]
    neg = scores[labels01 == 0]
    p, n = int(pos.size), int(neg.size)
    if p == 0 or n == 0:
        return float("nan")
    ranks = pd.Series(np.concatenate([pos, neg])).rank(method="average").to_numpy(dtype=np.float64)
    y = np.concatenate([np.ones(p, dtype=np.int8), np.zeros(n, dtype=np.int8)])
    auc = (np.sum(ranks[y == 1]) - p * (p + 1) / 2.0) / (p * n)
    return float(auc)


def _mean_or_nan(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    return float(np.mean(x))


def _compute_seed_metrics(df: pd.DataFrame, coverages: Sequence[float]) -> Tuple[Dict[str, float], Dict[str, str]]:
    label_col = _first_existing_column(df, LABEL_CANDIDATES)
    sem_col = _first_existing_column(df, SEM_SCORE_CANDIDATES)
    vis_col = _first_existing_column(df, VIS_SCORE_CANDIDATES)
    final_col = _first_existing_column(df, FINAL_SCORE_CANDIDATES)
    img_col = _first_existing_column(df, PATH_CANDIDATES)

    if label_col is None or sem_col is None or vis_col is None:
        raise ValueError("Missing required label/semantic/visual columns")

    work = pd.DataFrame()
    work["image_path"] = df[img_col].astype(str) if img_col is not None else np.arange(len(df)).astype(str)
    work["label"] = _to_label01(df[label_col])
    work["semantic_score"] = pd.to_numeric(df[sem_col], errors="coerce")
    work["visual_score"] = pd.to_numeric(df[vis_col], errors="coerce")
    if final_col is not None:
        work["final_score"] = pd.to_numeric(df[final_col], errors="coerce")
    else:
        work["final_score"] = np.nan

    # Recompute final score where missing.
    miss = work["final_score"].isna()
    if miss.any():
        a = work.loc[miss, "semantic_score"].to_numpy(dtype=np.float64)
        b = work.loc[miss, "visual_score"].to_numpy(dtype=np.float64)
        work.loc[miss, "final_score"] = (2.0 * a * b) / (a + b + EPS)

    work = work.dropna(subset=["label", "semantic_score", "visual_score", "final_score"]).copy()
    work["label"] = work["label"].astype(int)
    work = work[(work["label"] == 0) | (work["label"] == 1)]
    if work.empty:
        raise ValueError("No valid rows after standardization")

    sem = work["semantic_score"].to_numpy(dtype=np.float64)
    vis = work["visual_score"].to_numpy(dtype=np.float64)
    final = work["final_score"].to_numpy(dtype=np.float64)
    y = work["label"].to_numpy(dtype=np.int8)

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    p, n = int(pos_idx.size), int(neg_idx.size)
    if p == 0 or n == 0:
        raise ValueError("Need both anomaly and normal samples")

    sem_pos, sem_neg = sem[pos_idx], sem[neg_idx]
    vis_pos, vis_neg = vis[pos_idx], vis[neg_idx]
    final_pos, final_neg = final[pos_idx], final[neg_idx]

    z_sem = sem_pos[:, None] > sem_neg[None, :]
    z_vis = vis_pos[:, None] > vis_neg[None, :]
    flip = (z_sem != z_vis)
    flip_f = flip.astype(np.float64)

    error = (final_pos[:, None] <= final_neg[None, :]).astype(np.float64)

    sample_instability = np.zeros(len(work), dtype=np.float64)
    sample_error = np.zeros(len(work), dtype=np.float64)
    sample_instability[pos_idx] = np.mean(flip_f, axis=1)
    sample_instability[neg_idx] = np.mean(flip_f, axis=0)
    sample_error[pos_idx] = np.mean(error, axis=1)
    sample_error[neg_idx] = np.mean(error, axis=0)

    auc = _auc_from_scores(final, y)
    setting_inst_flip = float(np.mean(flip_f))
    setting_inst_var = 0.25 * setting_inst_flip
    setting_error = float(np.mean(error))

    risk_cov: Dict[str, float] = {}
    warnings: Dict[str, str] = {}
    order_desc = np.argsort(-sample_instability, kind="mergesort")
    total = len(work)
    for cov in coverages:
        reject_n = int(np.floor((1.0 - float(cov)) * total))
        keep_n = total - reject_n
        keep_idx = np.sort(order_desc[reject_n:reject_n + keep_n])
        keep_labels = y[keep_idx]
        if np.sum(keep_labels == 1) == 0 or np.sum(keep_labels == 0) == 0:
            risk_cov[f"risk_cov_{cov:.1f}"] = float("nan")
            warnings[f"risk_cov_{cov:.1f}"] = "accepted set lacks positive or negative"
            continue
        keep_scores = final[keep_idx]
        kp = keep_scores[keep_labels == 1]
        kn = keep_scores[keep_labels == 0]
        risk_cov[f"risk_cov_{cov:.1f}"] = float(np.mean(kp[:, None] <= kn[None, :]))

    metrics: Dict[str, float] = {
        "n_images": float(len(work)),
        "n_anomaly": float(np.sum(y == 1)),
        "n_normal": float(np.sum(y == 0)),
        "auroc": auc,
        "setting_instability_flip": setting_inst_flip,
        "setting_instability_var": setting_inst_var,
        "setting_error": setting_error,
    }
    metrics.update(risk_cov)
    return metrics, warnings


def _safe_float(v: object) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def _pick_argmax(df: pd.DataFrame, col: str) -> pd.Series:
    g = df.sort_values([col, "setting_instability_flip", "seed"], ascending=[False, True, True], kind="mergesort")
    return g.iloc[0]


def _pick_argmin(df: pd.DataFrame, col: str) -> pd.Series:
    g = df.sort_values([col, "auroc", "seed"], ascending=[True, False, True], kind="mergesort")
    return g.iloc[0]


def run(args: argparse.Namespace) -> None:
    promptad_root = os.path.abspath(args.promptad_root)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    coverages = tuple(float(x) for x in args.coverages)
    epsilons = tuple(float(x) for x in args.epsilons)

    csvs = _discover_per_sample_csvs(promptad_root)
    reason_counter: Counter[str] = Counter()
    rows: List[Dict[str, object]] = []
    warnings_rows: List[Dict[str, str]] = []
    seen_keys: set[Tuple[str, str, int, int]] = set()

    for p in csvs:
        parsed = _parse_setting_seed_from_path(p)
        if parsed is None:
            reason_counter["unparsed_setting_seed"] += 1
            continue
        key = (parsed.dataset, parsed.cls, parsed.k, parsed.seed)
        if key in seen_keys:
            reason_counter["duplicate_seed_setting"] += 1
            continue
        seen_keys.add(key)
        try:
            df = pd.read_csv(p)
            metrics, warns = _compute_seed_metrics(df, coverages=coverages)
        except Exception as e:
            reason_counter[f"seed_compute_error:{type(e).__name__}"] += 1
            continue
        row = {
            "dataset": parsed.dataset,
            "class": parsed.cls,
            "k": parsed.k,
            "seed": parsed.seed,
            "n_images": int(metrics["n_images"]),
            "n_anomaly": int(metrics["n_anomaly"]),
            "n_normal": int(metrics["n_normal"]),
            "auroc": metrics["auroc"],
            "setting_instability_flip": metrics["setting_instability_flip"],
            "setting_instability_var": metrics["setting_instability_var"],
            "setting_error": metrics["setting_error"],
        }
        for cov in coverages:
            row[f"risk_cov_{cov:.1f}"] = metrics.get(f"risk_cov_{cov:.1f}", float("nan"))
        rows.append(row)
        for ck, msg in warns.items():
            warnings_rows.append(
                {
                    "dataset": parsed.dataset,
                    "class": parsed.cls,
                    "k": str(parsed.k),
                    "seed": str(parsed.seed),
                    "coverage_key": ck,
                    "warning": msg,
                    "source_csv": p,
                }
            )

    per_seed = pd.DataFrame(rows)
    if per_seed.empty:
        raise SystemExit("No valid per-sample CSV found for analysis.")
    per_seed = per_seed.sort_values(["dataset", "class", "k", "seed"], kind="mergesort").reset_index(drop=True)
    per_seed_path = os.path.join(out_dir, "per_seed_metrics.csv")
    per_seed.to_csv(per_seed_path, index=False)

    summary_by_eps: Dict[float, pd.DataFrame] = {}
    agg_rows: List[Dict[str, object]] = []
    skipped_settings: List[Dict[str, object]] = []

    group_cols = ["dataset", "class", "k"]
    grouped = list(per_seed.groupby(group_cols, sort=True))

    for eps in epsilons:
        sel_rows: List[Dict[str, object]] = []
        for (dataset, cls, k), g in grouped:
            g = g.copy()
            if len(g) < 2:
                skipped_settings.append(
                    {"dataset": dataset, "class": cls, "k": int(k), "reason": "less_than_two_seeds", "epsilon": eps}
                )
                continue
            if np.allclose(g["setting_instability_flip"].to_numpy(dtype=np.float64), 0.0):
                skipped_settings.append(
                    {"dataset": dataset, "class": cls, "k": int(k), "reason": "all_seed_instability_zero", "epsilon": eps}
                )
                continue

            seed_auc_row = _pick_argmax(g, "auroc")
            best_auc = float(seed_auc_row["auroc"])
            candidate = g[g["auroc"] >= (best_auc - eps)].copy()
            if candidate.empty:
                skipped_settings.append(
                    {"dataset": dataset, "class": cls, "k": int(k), "reason": "empty_candidate_set", "epsilon": eps}
                )
                continue
            seed_stable_row = _pick_argmin(candidate, "setting_instability_flip")

            for cov in coverages:
                risk_col = f"risk_cov_{cov:.1f}"
                gg = g[np.isfinite(g[risk_col].to_numpy(dtype=np.float64))].copy()
                if gg.empty:
                    skipped_settings.append(
                        {
                            "dataset": dataset,
                            "class": cls,
                            "k": int(k),
                            "reason": f"no_valid_risk_{cov:.1f}",
                            "epsilon": eps,
                        }
                    )
                    continue
                oracle_row = _pick_argmin(gg, risk_col)
                sa, ss, so = int(seed_auc_row["seed"]), int(seed_stable_row["seed"]), int(oracle_row["seed"])
                ra = _safe_float(seed_auc_row[risk_col])
                rs = _safe_float(seed_stable_row[risk_col])
                ro = _safe_float(oracle_row[risk_col])
                if not np.isfinite(ra) or not np.isfinite(rs) or not np.isfinite(ro):
                    skipped_settings.append(
                        {
                            "dataset": dataset,
                            "class": cls,
                            "k": int(k),
                            "reason": f"nan_risk_{cov:.1f}",
                            "epsilon": eps,
                        }
                    )
                    continue
                sel_rows.append(
                    {
                        "dataset": dataset,
                        "class": cls,
                        "k": int(k),
                        "coverage": float(cov),
                        "epsilon": float(eps),
                        "seed_auc": sa,
                        "seed_stable": ss,
                        "seed_oracle": so,
                        "auroc_auc_seed": float(seed_auc_row["auroc"]),
                        "auroc_stable_seed": float(seed_stable_row["auroc"]),
                        "instability_auc_seed": float(seed_auc_row["setting_instability_flip"]),
                        "instability_stable_seed": float(seed_stable_row["setting_instability_flip"]),
                        "risk_auc_seed": ra,
                        "risk_stable_seed": rs,
                        "risk_oracle_seed": ro,
                        "delta_risk_auc_minus_stable": ra - rs,
                        "dist_auc_to_oracle": ra - ro,
                        "dist_stable_to_oracle": rs - ro,
                        "stable_improves": bool((ra - rs) > 0.0),
                    }
                )

        sel_df = pd.DataFrame(sel_rows)
        sel_df = sel_df.sort_values(["dataset", "class", "k", "coverage"], kind="mergesort").reset_index(drop=True)
        summary_by_eps[eps] = sel_df
        eps_tag = f"{eps:.3f}"
        sel_df.to_csv(os.path.join(out_dir, f"selection_summary_epsilon_{eps_tag}.csv"), index=False)

        if not sel_df.empty:
            for cov, g_cov in sel_df.groupby("coverage", sort=True):
                delta = g_cov["delta_risk_auc_minus_stable"].to_numpy(dtype=np.float64)
                d_ao = g_cov["dist_auc_to_oracle"].to_numpy(dtype=np.float64)
                d_so = g_cov["dist_stable_to_oracle"].to_numpy(dtype=np.float64)
                agg_rows.append(
                    {
                        "epsilon": float(eps),
                        "coverage": float(cov),
                        "n_settings_valid": int(len(g_cov)),
                        "mean_delta_risk": float(np.mean(delta)),
                        "median_delta_risk": float(np.median(delta)),
                        "win_rate_stable_over_auc": float(np.mean(delta > 0)),
                        "mean_dist_auc_to_oracle": float(np.mean(d_ao)),
                        "mean_dist_stable_to_oracle": float(np.mean(d_so)),
                        "oracle_gap_reduction": float(np.mean(d_ao) - np.mean(d_so)),
                    }
                )

    agg_df = pd.DataFrame(agg_rows).sort_values(["epsilon", "coverage"], kind="mergesort")
    agg_df.to_csv(os.path.join(out_dir, "aggregate_summary.csv"), index=False)

    clean_rows: List[pd.DataFrame] = []
    for eps, sel_df in summary_by_eps.items():
        if sel_df.empty:
            continue
        cond = (
            (np.abs(sel_df["auroc_auc_seed"] - sel_df["auroc_stable_seed"]) <= eps)
            & ((sel_df["risk_auc_seed"] - sel_df["risk_stable_seed"]) > 0)
            & (sel_df["instability_auc_seed"] > sel_df["instability_stable_seed"])
        )
        clean_rows.append(sel_df[cond].copy())
    clean_df = pd.concat(clean_rows, axis=0, ignore_index=True) if clean_rows else pd.DataFrame()
    if not clean_df.empty:
        clean_df = clean_df.sort_values("delta_risk_auc_minus_stable", ascending=False, kind="mergesort").head(20)
    clean_df.to_csv(os.path.join(out_dir, "clean_cases.csv"), index=False)

    # Figure 1: best clean case scatter.
    if not clean_df.empty:
        case = clean_df.sort_values("delta_risk_auc_minus_stable", ascending=False, kind="mergesort").iloc[0]
        cov = float(case["coverage"])
        eps = float(case["epsilon"])
        s_key = (case["dataset"], case["class"], int(case["k"]))
        g = per_seed[
            (per_seed["dataset"] == s_key[0]) & (per_seed["class"] == s_key[1]) & (per_seed["k"] == s_key[2])
        ].copy()
        risk_col = f"risk_cov_{cov:.1f}"
        fig, ax = plt.subplots(figsize=(7, 5))
        sc = ax.scatter(
            g["auroc"].to_numpy(dtype=np.float64),
            g[risk_col].to_numpy(dtype=np.float64),
            c=g["setting_instability_flip"].to_numpy(dtype=np.float64),
            cmap="viridis",
            s=70,
            edgecolors="k",
            linewidths=0.5,
        )
        for _, r in g.iterrows():
            ax.text(float(r["auroc"]) + 1e-4, float(r[risk_col]) + 1e-4, str(int(r["seed"])), fontsize=8)

        def _mark(seed: int, marker: str, label: str, color: str) -> None:
            rr = g[g["seed"] == seed].iloc[0]
            ax.scatter(
                [float(rr["auroc"])],
                [float(rr[risk_col])],
                marker=marker,
                s=144,
                facecolors="none",
                edgecolors=color,
                linewidths=2,
                label=label,
            )

        _mark(int(case["seed_auc"]), "s", "AUROC-selected", "red")
        _mark(int(case["seed_stable"]), "D", "Instability-selected", "blue")
        _mark(int(case["seed_oracle"]), "*", "Oracle", "orange")
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("setting_instability_flip")
        ax.set_xlabel("AUROC")
        ax.set_ylabel(f"Decision risk @ coverage={cov:.1f}")
        ax.set_title(f"{s_key[0]} / {s_key[1]} / k={s_key[2]} (epsilon={eps:.3f})")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", markerscale=0.8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "selection_case_scatter.png"), dpi=180)
        plt.close(fig)

    # Figure 2: aggregate mean delta risk bar.
    if not agg_df.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        coverages_sorted = sorted(agg_df["coverage"].unique().tolist(), reverse=True)
        eps_sorted = sorted(agg_df["epsilon"].unique().tolist())
        x = np.arange(len(coverages_sorted), dtype=np.float64)
        width = 0.35 if len(eps_sorted) == 2 else max(0.8 / max(1, len(eps_sorted)), 0.2)
        for i, eps in enumerate(eps_sorted):
            ys = []
            sub = agg_df[agg_df["epsilon"] == eps]
            for c in coverages_sorted:
                r = sub[sub["coverage"] == c]
                ys.append(float(r["mean_delta_risk"].iloc[0]) if not r.empty else np.nan)
            shift = (i - (len(eps_sorted) - 1) / 2.0) * width
            ax.bar(x + shift, ys, width=width, label=f"epsilon={eps:.3f}")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{c:.1f}" for c in coverages_sorted])
        ax.set_xlabel("Coverage")
        ax.set_ylabel("mean_delta_risk (risk_auc - risk_stable)")
        ax.set_title("Aggregate Delta Risk")
        ax.axhline(0.0, color="black", linewidth=1)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "aggregate_delta_risk.png"), dpi=180)
        plt.close(fig)

    skip_df = pd.DataFrame(skipped_settings)
    if not skip_df.empty:
        skip_df.to_csv(os.path.join(out_dir, "skipped_settings.csv"), index=False)
    if warnings_rows:
        pd.DataFrame(warnings_rows).to_csv(os.path.join(out_dir, "warnings.csv"), index=False)

    print(f"total_seed_csvs_discovered={len(csvs)}")
    print(f"valid_seed_rows={len(per_seed)}")
    print(f"total_settings={per_seed.groupby(['dataset', 'class', 'k']).ngroups}")
    print("skip_reason_counts:")
    reason_from_selection = defaultdict(int)
    for x in skipped_settings:
        reason_from_selection[str(x["reason"])] += 1
    for k, v in sorted(reason_counter.items()):
        print(f"  {k}: {v}")
    for k, v in sorted(reason_from_selection.items()):
        print(f"  selection_{k}: {v}")
    if not agg_df.empty:
        print("coverage_winrate_mean_delta:")
        for _, r in agg_df.sort_values(["epsilon", "coverage"]).iterrows():
            print(
                "  "
                f"epsilon={float(r['epsilon']):.3f}, coverage={float(r['coverage']):.1f}, "
                f"win_rate={float(r['win_rate_stable_over_auc']):.4f}, "
                f"mean_delta={float(r['mean_delta_risk']):.6f}"
            )
    print(f"wrote={out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PromptAD pilot experiment: AUROC-only vs instability-aware selection.")
    p.add_argument(
        "--promptad-root",
        type=str,
        default=os.path.join(os.path.dirname(__file__), ".."),
        help="PromptAD project root",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "result_analysis", "pilot_instability_selection"),
        help="Output directory",
    )
    p.add_argument("--coverages", nargs="+", type=float, default=list(DEFAULT_COVERAGES))
    p.add_argument("--epsilons", nargs="+", type=float, default=list(DEFAULT_EPSILONS))
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
