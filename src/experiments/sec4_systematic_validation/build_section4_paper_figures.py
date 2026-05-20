#!/usr/bin/env python3
"""
Build Section 4 PromptAD paper figures under ``outputs/figures/section4/``.

All conditioned statistics use the near-AUROC **seed-pair** universe (see
``sec4_near_auroc_universe.py``). No global pooled pairwise averaging.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.experiments.sec4_systematic_validation.paper_plotting import (
    plot_fig2_same_auroc,
    plot_fig3_4_merged,
    plot_fig5_6_merged,
    plot_fig7_aggregate_delta_risk,
)
from src.experiments.sec4_systematic_validation.sec4_near_auroc_universe import (
    DEFAULT_COVERAGES_FIG56_REGIME,
    DEFAULT_COVERAGES_FIG7,
    DEFAULT_EPSILON,
    DEFAULT_EPSILONS_FIG7,
    DEFAULT_TAU,
    aggregate_fig7,
    build_near_auroc_pair_cases,
    build_risk_lookup,
    build_setting_signals,
    conditioned_margin_tables,
    conditioned_pairwise_margin_tables,
    load_pairwise_for_pair_universe,
    failure_conditioned_signals_from_cases,
    failure_gate_from_cases,
    assign_setting_global_roles,
    pick_fig2_setting,
    weighted_regime_failure_rates,
)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(type(obj))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path, default=Path("."))
    ap.add_argument("--cache-dir", type=Path, default=Path("outputs/cached_results/sec4_systematic"))
    ap.add_argument("--sec3-cache-dir", type=Path, default=Path("outputs/cached_results/sec3_promptad"))
    ap.add_argument("--aggregate-dir", type=Path, default=Path("outputs/cached_results/promptad_pairwise"))
    ap.add_argument(
        "--raw-wide",
        type=Path,
        default=Path("outputs/cached_results/raw_scores/promptad/unified_raw_scores_wide.csv"),
        help="Wide raw for gap/score_var/margin on AUROC-selected seeds (cached export only).",
    )
    ap.add_argument("--risk-coverage-csv", type=Path, default=None)
    ap.add_argument("--fig-dir", type=Path, default=Path("outputs/figures/section4"))
    ap.add_argument("--paper-csv-dir", type=Path, default=Path("outputs/cached_results/sec4_systematic/paper"))
    ap.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    ap.add_argument("--tau", type=float, default=DEFAULT_TAU)
    args = ap.parse_args()

    repo = args.repo_root.resolve()
    cache = args.cache_dir.resolve()
    fig_dir = args.fig_dir.resolve()
    paper_csv = args.paper_csv_dir.resolve()
    paper_csv.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    setting_path = cache / "setting_level_metrics.csv"
    if not setting_path.is_file():
        setting_path = args.aggregate_dir.resolve() / "setting_level_metrics.csv"
    if not setting_path.is_file():
        raise FileNotFoundError("setting_level_metrics.csv")

    setting_df = pd.read_csv(setting_path)
    rc_path = args.risk_coverage_csv
    if rc_path is None:
        for cand in (
            args.sec3_cache_dir.resolve() / "risk_coverage.csv",
            args.aggregate_dir.resolve() / "risk_coverage.csv",
            cache / "risk_coverage.csv",
        ):
            if cand.is_file():
                rc_path = cand
                break
    if rc_path is None or not Path(rc_path).is_file():
        raise FileNotFoundError("risk_coverage.csv")
    rc_path = Path(rc_path).resolve()
    rc_df = pd.read_csv(rc_path)
    rc_lookup = build_risk_lookup(rc_df)

    wide_candidates = [
        args.raw_wide.resolve(),
        repo / "outputs/cached_results/raw_scores/promptad/unified_raw_scores_wide.csv",
    ]
    wide_path = next((p for p in wide_candidates if p.is_file()), None)
    cm_detail_path = cache / "controlled_margin_detail.csv"
    if not cm_detail_path.is_file():
        cm_detail_path = args.aggregate_dir.resolve() / "controlled_margin_detail.csv"

    cases = build_near_auroc_pair_cases(
        setting_df,
        rc_lookup,
        epsilon=float(args.epsilon),
        tau=float(args.tau),
        coverages=DEFAULT_COVERAGES_FIG7,
        epsilons=DEFAULT_EPSILONS_FIG7,
    )
    if cases.empty:
        raise SystemExit("near-AUROC pair universe is empty — check caches and epsilon")

    cases.to_csv(paper_csv / "near_auroc_pair_cases.csv", index=False)
    agg7 = aggregate_fig7(cases)
    agg7.to_csv(paper_csv / "aggregate_summary_pairs.csv", index=False)

    # Regime / failure-rate panels: practical coverages (paper strengthening uses 0.5/0.7/0.8).
    regime_coverages = list(DEFAULT_COVERAGES_FIG56_REGIME)
    gate = failure_gate_from_cases(
        cases, epsilon=float(args.epsilon), coverages=regime_coverages
    )
    gate.to_csv(paper_csv / "failure_gate_analysis.csv", index=False)

    signal_df, sig_warn = build_setting_signals(setting_df, cases, wide_path)
    signal_df.to_csv(paper_csv / "setting_signals_auroc_seed.csv", index=False)

    fc = failure_conditioned_signals_from_cases(
        cases, signal_df, epsilon=float(args.epsilon), coverages=DEFAULT_COVERAGES_FIG56_REGIME
    )
    fc.to_csv(paper_csv / "failure_conditioned_signal_analysis.csv", index=False)

    chain, cm_cond = pd.DataFrame(), pd.DataFrame()
    margin_note = ""
    n_conditioned_pair_rows = 0
    pairwise_metrics_path = cache / "pairwise_metrics.csv"
    if not pairwise_metrics_path.is_file():
        pairwise_metrics_path = args.aggregate_dir.resolve() / "pairwise_metrics.csv"
    if not pairwise_metrics_path.is_file():
        pairwise_metrics_path = args.sec3_cache_dir.resolve() / "pairwise_metrics.csv"

    if pairwise_metrics_path.is_file():
        pairwise_df = load_pairwise_for_pair_universe(
            pairwise_metrics_path, cases, epsilon=float(args.epsilon)
        )
        n_conditioned_pair_rows = int(len(pairwise_df))
        chain, cm_cond = conditioned_pairwise_margin_tables(
            pairwise_df, cases, epsilon=float(args.epsilon)
        )
        chain.to_csv(paper_csv / "mechanism_chain_summary.csv", index=False)
        cm_cond.to_csv(paper_csv / "controlled_margin_analysis_conditioned.csv", index=False)
        margin_note = (
            f"strict pairwise rows from {pairwise_metrics_path.name} "
            f"({len(pairwise_df)} conditioned rows), filtered to seeds in pair universe"
        )
    elif cm_detail_path.is_file():
        chain, cm_cond = conditioned_margin_tables(
            pd.read_csv(cm_detail_path), cases, epsilon=float(args.epsilon)
        )
        chain.to_csv(paper_csv / "mechanism_chain_summary.csv", index=False)
        cm_cond.to_csv(paper_csv / "controlled_margin_analysis_conditioned.csv", index=False)
        margin_note = "FALLBACK: settings in pair universe only (per-setting margin tertiles, not pair-filtered pairwise rows)"
    else:
        margin_note = "controlled_margin_detail.csv and pairwise_metrics.csv missing — fig3_4 (a)(b) skipped"

    y_regime = weighted_regime_failure_rates(gate)

    gen_time = datetime.now(timezone.utc).isoformat()

    # --- Fig2 ---
    fig2_summary: Dict[str, Any] = {
        "figure": "fig2_same_auroc",
        "generation_time": gen_time,
        "setting_selection": {
            "universe": f"near-AUROC seed pairs with |ΔAUROC| < {args.epsilon}",
            "failure_criterion": f"pair-level Δrisk > {args.tau} and instability_AUROC > instability_inst (for ranking only)",
            "aggregation_unit": "one (dataset, category, shot, coverage) with strongest pair failure",
        },
        "role_definition": {
            "scope": "setting-global over all seeds in the chosen setting",
            "auroc_selected": "argmax setting_auroc (tie: smaller seed)",
            "instability_selected": "argmin mean_pairwise_instability (tie: smaller seed)",
            "oracle": "argmin decision risk @ coverage (tie: smaller seed)",
        },
    }
    try:
        selection = pick_fig2_setting(cases, epsilon=float(args.epsilon), tau=float(args.tau))
        roles = assign_setting_global_roles(
            setting_df,
            rc_lookup,
            dataset=str(selection["dataset"]),
            category=str(selection["category"]),
            shot=int(selection["shot"]),
            coverage=float(selection["coverage"]),
        )
        plot_fig2_same_auroc(selection, roles, setting_df, rc_df, fig_dir / "fig2_same_auroc")
        fig2_summary["selected_setting"] = {
            "dataset": selection["dataset"],
            "category": selection["category"],
            "shot": int(selection["shot"]),
            "coverage": float(selection["coverage"]),
            "epsilon": float(selection["epsilon"]),
            "exemplar_delta_risk_pair": float(selection["exemplar_delta_risk_pair"]),
        }
        fig2_summary["setting_global_roles"] = roles
    except ValueError as e:
        fig2_summary["error"] = str(e)

    (fig_dir / "fig2_same_auroc_summary.json").write_text(
        json.dumps(fig2_summary, indent=2, default=_json_default), encoding="utf-8"
    )

    # --- Fig3-4 ---
    fig34_summary: Dict[str, Any] = {
        "figure": "fig3_4_merged_mechanism",
        "generation_time": gen_time,
        "epsilon": float(args.epsilon),
        "tau": float(args.tau),
        "panel_a_b": {
            "conditioning": "strict pairwise-level on anomaly–normal pair rows",
            "margin_bucket": "tertiles of abs(fused_margin) on conditioned pairwise rows",
            "instability_split": "bottom / top third of pairwise_instability within each margin bucket",
            "error_metric": "mean pairwise_error (stable vs unstable tails)",
            "note": margin_note,
            "pairwise_metrics_path": str(pairwise_metrics_path) if pairwise_metrics_path.is_file() else None,
        },
        "panel_c": {
            "statistical_universe": f"near-AUROC seed pairs, |ΔAUROC| < {args.epsilon}, coverages={list(DEFAULT_COVERAGES_FIG56_REGIME)}",
            "failure_definition": f"risk(AUROC-selected) - risk(instability-selected) > {args.tau}",
            "instability_regime": "tertiles of max(instability_auc_seed, instability_inst_seed) on pair rows (per coverage, then weighted)",
            "weighted_failure_rate": {
                "low_I": float(y_regime[0]) if np.isfinite(y_regime[0]) else None,
                "mid_I": float(y_regime[1]) if np.isfinite(y_regime[1]) else None,
                "high_I": float(y_regime[2]) if np.isfinite(y_regime[2]) else None,
            },
        },
    }
    if not chain.empty and not cm_cond.empty:
        fig34_summary["panel_a_b"]["chain"] = chain.to_dict(orient="records")
        fig34_summary["panel_a_b"]["controlled_margin"] = cm_cond.to_dict(orient="records")
        fig34_summary["panel_a_b"]["n_conditioned_pair_rows"] = n_conditioned_pair_rows
        plot_fig3_4_merged(chain, cm_cond, y_regime, fig_dir / "fig3_4_merged_mechanism")
    else:
        fig34_summary["skipped"] = "missing conditioned margin tables"
    (fig_dir / "fig3_4_merged_mechanism_summary.json").write_text(
        json.dumps(fig34_summary, indent=2, default=_json_default), encoding="utf-8"
    )
    (cache / "fig3_4_mechanism_summary.json").write_text(
        json.dumps(fig34_summary, indent=2, default=_json_default), encoding="utf-8"
    )

    # --- Fig5-6 ---
    fig56_summary: Dict[str, Any] = {
        "figure": "fig5_6_merged_failure",
        "generation_time": gen_time,
        "statistical_universe": fig34_summary["panel_c"]["statistical_universe"],
        "failure_definition": fig34_summary["panel_c"]["failure_definition"],
        "signals": {
            "instability": "mean_pairwise_instability of AUROC-selected seed",
            "gap": "mean |semantic-visual| per image from wide raw",
            "score_var": "variance of fused score from wide raw",
            "margin": "mean |fused_anomaly - fused_normal| from wide raw",
        },
        "signal_warnings": sig_warn,
        "failure_conditioned_rows": fc.to_dict(orient="records") if not fc.empty else [],
    }
    if fc.empty or not np.all(np.isfinite(y_regime)):
        fig56_summary["warning"] = "incomplete signal or regime data"
    else:
        plot_fig5_6_merged(fc, y_regime, fig_dir / "fig5_6_merged_failure")
    (fig_dir / "fig5_6_merged_failure_summary.json").write_text(
        json.dumps(fig56_summary, indent=2, default=_json_default), encoding="utf-8"
    )

    # --- Fig7 ---
    fig7_summary: Dict[str, Any] = {
        "figure": "fig7_delta_risk",
        "generation_time": gen_time,
        "statistical_universe": "near-AUROC seed pairs (all ε in plot)",
        "aggregation_unit": "mean delta_risk over pair rows per (epsilon, coverage)",
        "failure_definition": f"pair-level delta_risk = risk_AUROC - risk_instability; bars show mean (not failure rate)",
        "n_pair_rows": int(len(cases)),
        "aggregate_table": agg7.to_dict(orient="records"),
    }
    plot_fig7_aggregate_delta_risk(agg7, fig_dir / "fig7_delta_risk")
    (fig_dir / "fig7_delta_risk_summary.json").write_text(
        json.dumps(fig7_summary, indent=2, default=_json_default), encoding="utf-8"
    )

    master = {
        "generation_time": gen_time,
        "epsilon": float(args.epsilon),
        "tau": float(args.tau),
        "n_near_auroc_pair_rows": int(len(cases)),
        "n_unique_pairs": int(cases[["dataset", "category", "shot", "seed_a", "seed_b"]].drop_duplicates().shape[0]),
        "inputs": {
            "setting_level_metrics": str(setting_path),
            "risk_coverage": str(rc_path),
            "wide_raw": str(wide_path) if wide_path else None,
            "controlled_margin_detail": str(cm_detail_path) if cm_detail_path.is_file() else None,
        },
        "paper_csv_dir": str(paper_csv),
        "figures": {
            "fig2": str(fig_dir / "fig2_same_auroc.png"),
            "fig3_4": str(fig_dir / "fig3_4_merged_mechanism.png"),
            "fig5_6": str(fig_dir / "fig5_6_merged_failure.png"),
            "fig7": str(fig_dir / "fig7_delta_risk.png"),
        },
    }
    (cache / "section4_paper_figures_summary.json").write_text(
        json.dumps(master, indent=2, default=_json_default), encoding="utf-8"
    )

    audit_path = fig_dir / "SECTION4_FIGURE_AUDIT.md"
    _write_audit_md(audit_path, master, fig2_summary, fig34_summary, fig56_summary, fig7_summary, sig_warn, margin_note)

    print(json.dumps(master, indent=2, default=_json_default))
    return 0


def _write_audit_md(
    path: Path,
    master: Dict[str, Any],
    fig2: Dict[str, Any],
    fig34: Dict[str, Any],
    fig56: Dict[str, Any],
    fig7: Dict[str, Any],
    sig_warn: List[str],
    margin_note: str,
) -> None:
    path.write_text(
        f"""# Section 4 figure audit (PromptAD)

Generated: {master["generation_time"]}

## Unified statistical universe

- **Candidate set** \\(\\mathcal{{C}}\\): unordered seed pairs \\((A,B)\\) within the same
  `(dataset, category, shot)` with \\(|\\mathrm{{AUROC}}_A - \\mathrm{{AUROC}}_B| < \\varepsilon\\),
  \\(\\varepsilon = {master["epsilon"]}\\) (default, matches pilot / mechanism scripts).
- **Roles** (per pair, tie-break: smaller `seed` id):
  - AUROC-selected: higher `setting_auroc`
  - instability-selected: lower `mean_pairwise_instability`
- **Risk**: `mean_pairwise_risk` from `risk_coverage.csv` at coverage \\(c\\) (interpolated on the coverage grid).
- **Failure** (conditioned analysis): \\(\\mathrm{{risk}}_{{\\mathrm{{AUROC}}}} - \\mathrm{{risk}}_{{\\mathrm{{inst}}}} > \\tau\\)
  with \\(\\tau = {master["tau"]}\\) (matches `mechanism_driven_analysis.py` `failure_s2`: `delta_risk > 0.01`).

**Not used for these figures:** global pooled `failure_conditioned_signal_comparison.csv`, all-pairs
`instability_regime_failure_rate.csv`, or `decision_consequence_delta_risk.csv` (pair-level oracle contrast).

**Upstream nuance:** `failure_driven_analysis.py` labels failure as `gap_auc = risk_AUROC - risk_oracle > 0.01`
(still threshold 0.01, different contrast). Section 4 paper bundle here follows **Δrisk vs instability-selected**.

Rows in universe: **{master["n_near_auroc_pair_rows"]}** (pair × coverage × epsilon);
**{master["n_unique_pairs"]}** unique seed pairs at ε={master["epsilon"]}.

---

## What was wrong before (aggregation drift)

| Issue | Old behavior | Fix |
|-------|----------------|-----|
| Universe | All settings in near-AUROC **pool** (max AUROC − ε) | Explicit **seed pairs** with \\|ΔAUROC\\| < ε |
| Fig5 (a) | Global `failure_conditioned_signal_comparison` or supplementary CSV (81-setting failure analysis) | Signals on **AUROC-selected seed** of each pair row; failure from Δrisk |
| Fig5–6 (b) | All settings / weak tertiles on pooled selection | Tertiles of `instability_auc_seed` on **pair rows**, weighted failure gate |
| Fig3–4 (c) | `decision_consequence` setting tertiles → flat line | `failure_gate` from pair universe |
| Fig7 | Histogram of oracle Δrisk or diluted setting means | **Mean** `risk_AUROC − risk_instability` over pair rows vs coverage (pilot aggregate) |
| Fig3–4 (a)(b) | Global `controlled_margin_analysis` | Settings that appear in \\(\\mathcal{{C}}\\) only ({margin_note}) |

---

## Per-figure summary

### Fig2 (`fig2_same_auroc`)

| Field | Definition |
|-------|------------|
| Setting choice | From near-AUROC pair universe: exemplar with strongest pair-level failure (Δrisk > τ) |
| Plot | All seeds in that setting at chosen coverage |
| Markers | **Setting-global:** max AUROC (red), min instability (blue), min risk @ coverage (orange); ties → smaller seed |

{json.dumps({k: fig2.get(k) for k in ("selected_setting", "setting_global_roles", "error") if k in fig2}, indent=2, default=_json_default)}

### Fig3–4 (`fig3_4_merged_mechanism`)

| Panel | Aggregation unit | Notes |
|-------|------------------|-------|
| (a) Instability | **Pairwise rows** in \\(\\mathcal{{C}}\\) seeds; tertiles of \\|fused margin\\| | Mean `pairwise_instability` per bucket |
| (b) Error | Same pairwise-conditioned rows | low-I vs high-I tails on `pairwise_instability`; mean `pairwise_error` |
| (c) Failure | Pair rows, ε={master["epsilon"]}, cov ∈ {list(DEFAULT_COVERAGES_FIG56_REGIME)} | Regime = tertiles of `instability_auc_seed`; **weighted** failure rate |

Panel (c) rates: low={fig34["panel_c"]["weighted_failure_rate"]["low_I"]}, mid={fig34["panel_c"]["weighted_failure_rate"]["mid_I"]}, high={fig34["panel_c"]["weighted_failure_rate"]["high_I"]}.

**Fig3_4(a)(b) is now strictly pairwise-conditioned** because `pairwise_metrics.csv` is available: margin tertiles and instability/error splits are computed on filtered anomaly–normal **pair rows** (not setting-level proxies). See `{margin_note}`.

### Fig5–6 (`fig5_6_merged_failure`)

| Panel | Definition |
|-------|------------|
| (a) | Mean signal at **AUROC-selected** seed, failure vs non-failure pair rows (same universe) |
| (b) | Weighted failure rate by instability regime (same as Fig3–4c) |

Signals: instability from `setting_level_metrics`; gap / score_var / margin from **wide raw** scan (cached scores only, no Stage-1 rerun).
Warnings: {sig_warn or ["none"]}

### Fig7 (`fig7_delta_risk`)

| Field | Definition |
|-------|------------|
| Universe | All pair rows in \\(\\mathcal{{C}}\\) |
| Y-axis | Mean `risk_AUROC − risk_instability` per (epsilon, coverage) |
| Intent | Instability-aware selection **reduces** risk when mean > 0 at practical coverages |

---

## Reproduce

```bash
PYTHONPATH=. python3 src/experiments/sec4_systematic_validation/build_section4_paper_figures.py --repo-root .
```

Also wired in `scripts/rebuild_main_figures.sh` (Stage 3).
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
