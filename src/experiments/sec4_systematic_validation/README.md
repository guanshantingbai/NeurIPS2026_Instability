# Section 4 Systematic Validation

Two execution modes:

## A) From PromptAD unified raw (no train/infer, no `external/PromptAD` scripts)

If **`outputs/cached_results/raw_scores/promptad/unified_raw_scores_wide.csv`** or **`unified_raw_scores_long.csv`** exists, `run.sh` runs **`analyze_sec4_promptad_from_raw.py`** first and **exits** (stub `compute_*.py` chain is skipped).

**Pairwise setting:** `(dataset, category, shot, seed)` — all anomaly×normal pairs (optional **`--max-pairs-per-setting`** uniform subsample; recorded in `sec4_systematic_from_raw_summary.json`).

When using **`run.sh`**, you may set `SEC4_MAX_PAIRS_PER_SETTING` and `SEC4_PAIR_SAMPLING_SEED` (same semantics as the CLI flags above).

Per pair:

- `z_sem`, `z_vis`, `z_fused` (strict `>`; ties → 0)
- `pairwise_instability = Var([z_sem, z_vis])`
- `pairwise_error = 1 - z_fused`
- `fused_margin`, `abs_fused_margin`

**Setting-level AUROC** in `setting_level_metrics.csv` is **`mean(z_fused)`** over pairs (same convention as Section 3.1.1).

**Near-AUROC candidate pairs** (default **`--near-auroc-epsilon 0.002`**): same `dataset` / `category` / `shot`, different `seed`, with `|Δ setting_auroc| < ε`.

**Analyses (CSV + figures):**

| Deliverable | Cache CSV | Figure |
|-------------|-----------|--------|
| Controlled margin | `controlled_margin_analysis.csv` (+ detail `controlled_margin_detail.csv`) | `fig_controlled_margin.png` |
| Failure-conditioned signals | `failure_conditioned_signal_comparison.csv` | `fig_failure_conditioned_signals.png` |
| Instability regime failure rate | `instability_regime_failure_rate.csv` | `fig_instability_regime_failure_rate.png` |
| Decision consequence / delta risk | `decision_consequence_delta_risk.csv` | `fig_decision_consequence_delta_risk.png` |

**Controlled margin:** tertiles of `abs_fused_margin` (per setting when enough pairs, else **pooled** over all pairs). Within each margin bucket, **low vs high** instability subsets use the lowest / highest third of `pairwise_instability` ranks (disjoint tails). Aggregated table matches the spirit of upstream `controlled_margin_analysis.csv` (error in low-I vs high-I tails).

**Decision consequence:** for each near-AUROC seed pair, take the seed with **higher** `mean_pairwise_instability`; `delta_risk` = `delta_risk(mean_pairwise_error_that_seed, min(errors))` via `src.core.selection_failure.delta_risk`.

Optional CLI (not passed by default `run.sh`):

```bash
PYTHONPATH=. python3 src/experiments/sec4_systematic_validation/analyze_sec4_promptad_from_raw.py \
  --raw-dir outputs/cached_results/raw_scores/promptad \
  --near-auroc-epsilon 0.002 \
  --max-pairs-per-setting 500000 \
  --pair-sampling-seed 0
```

## B) Stub fast path (no unified raw)

If unified raw is **missing**, behavior is unchanged: `build_candidate_pairs.py`, `compute_*.py` placeholders, `plot_sec4_figures.py` markers, and optional **`FULL_RUN=1`** external strengthening (see `run.sh`).

## Run

```bash
bash src/experiments/sec4_systematic_validation/run.sh
```

## Config

`config.yaml` documents legacy defaults (`near_auroc_gap: 0.002`); the raw-driven analyzer uses argparse defaults aligned with that value.
