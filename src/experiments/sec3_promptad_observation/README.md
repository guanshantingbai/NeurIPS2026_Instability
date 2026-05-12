# Section 3.1.1 PromptAD Observation

Empirical summaries from **PromptAD Stage 1 unified raw scores** only (no training / inference from this `run.sh`).

## Definition (pairwise reproduction)

Analysis follows a **pairwise anomaly–normal** construction per setting `(dataset, category, shot, seed)`:

- For each anomaly image \(a\) and normal image \(n\), compare scores with strict `>` (ties \(\Rightarrow z=0\)):
  - \(z_{\mathrm{sem}} = \mathbb{1}[\mathrm{sem}(a) > \mathrm{sem}(n)]\), same for visual and fused.
  - **Pairwise instability** \(= \mathrm{Var}([z_{\mathrm{sem}}, z_{\mathrm{vis}}])\) (population variance over the two branches).
  - **Pairwise error** \(= 1 - z_{\mathrm{fused}}\).
  - **Fused margin** \(= \mathrm{fused}(a) - \mathrm{fused}(n)\).

Setting-level **AUROC** in the exported tables is **`mean(z_fused)`** over all pairs in that setting (pairwise win rate for fused; not sklearn ROC on pooled image labels).

**Risk–coverage** does **not** use a semantic–visual score-difference proxy. The accepted subset is built by ordering samples (default: ascending **sample mean pairwise instability**; optional `--risk-by fused` keeps high fused scores first via internal \(-\)fused ordering). Only images in the accepted subset are used to form anomaly–normal pairs; the curve reports **mean `pairwise_error`** on those pairs.

Optional **`--max-pairs-per-setting`** caps pairs per setting (uniform random subsample; seed `--pair-sampling-seed`). The summary JSON records whether subsampling was used.

When using **`src/experiments/sec3_promptad_observation/run.sh`**, you may set:

- `SEC3_PROMPTAD_MAX_PAIRS_PER_SETTING` → passed as `--max-pairs-per-setting`
- `SEC3_PROMPTAD_PAIR_SAMPLING_SEED` → passed as `--pair-sampling-seed`

## Inputs

- **Preferred:** `outputs/cached_results/raw_scores/promptad/unified_raw_scores_wide.csv`  
  or `unified_raw_scores_long.csv` (from `FULL_RUN=1 bash scripts/run_promptad_raw.sh` with `PROMPTAD_MODE=export`).
- **Optional stub:** set `SEC3_PROMPTAD_ALLOW_STUB=1` to copy bundled `pairwise_stub.csv` (non-empirical).

## Outputs

Under `outputs/cached_results/sec3_promptad/`:

- `pairwise_metrics.csv` — one row per anomaly–normal pair (or subsample).
- `setting_level_metrics.csv` — `setting_auroc`, `mean_pairwise_instability`, `mean_pairwise_error`, counts.
- `sample_level_metrics.csv` — per-image means over pairs where that image appears as anomaly or normal.
- `same_auroc_instability_pairs.csv` — seed pairs within `(dataset, category, shot)` with near-equal `setting_auroc` but large gap in mean pairwise instability.
- `risk_coverage.csv` — long format: one row per `(setting, coverage step)`; pooled figure interpolates these curves across settings.
- `sec3_promptad_from_raw_summary.json` — run metadata and sampling flags.

Figures under `outputs/figures/sec3_promptad/`: `scatter_auroc_vs_instability.png`, `hist_instability_distribution.png`, `scatter_instability_vs_ranking_error.png`, `risk_coverage_ranking_error.png`, and `scatter_same_auroc_instability_gaps.png` when applicable.

## Run

```bash
bash src/experiments/sec3_promptad_observation/run.sh
```

Optional analyzer flags (not passed by default `run.sh`):

```bash
python3 src/experiments/sec3_promptad_observation/analyze_sec3_promptad_from_raw.py \
  --raw-dir outputs/cached_results/raw_scores/promptad \
  --max-pairs-per-setting 500000 --pair-sampling-seed 0 --risk-by instability
```

If raw scores are missing, the script exits with instructions to run `scripts/run_promptad_raw.sh` first.
