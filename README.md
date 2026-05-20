# NeurIPS2026_Instability

Paper-section-oriented reproducibility repository for the NeurIPS 2026 instability paper.

## Design Principle

This repository is organized by paper sections and appendix tasks, not by model families.
Each section has explicit inputs, outputs, scripts, and commands.

## Repository Layout

- `docs/`: reproducibility guide, figure/table map, dataset and environment docs.
- `docs/PIPELINE_STAGES.md`: **canonical Stages 0–4** (preparation → evidence → aggregation → main figures → appendix).
- `scripts/`: top-level orchestration (`stage0_prepare.sh`, `run_*_raw.sh`, `rebuild_main_figures.sh`, …).
- `src/core/`: shared analysis logic (pairwise, instability, risk-coverage, metrics).
- `src/experiments/`: section-level experiment packages.
- `src/models/*_adapter/`: thin adapters to external model repositories.
- `external/`: third-party model code (PromptAD, PaDiM, PatchCore).
- `outputs/`: generated figures/tables/cached lightweight results.
- `figures/`: curated figures grouped by main paper vs appendix.

## Quick Start

1. Read `docs/ENVIRONMENT.md`, `docs/DATASET.md`, and **`docs/PIPELINE_STAGES.md`**.
2. **Stage 0 — preparation** (paths, `external/`, sanity; PatchCore weights check is a sub-step):
   ```bash
   bash scripts/stage0_prepare.sh
   ```
3. **Stage 1 — model evidence** (train/infer/export → `unified_raw_scores_*`):
   - `FULL_RUN=1 bash scripts/run_promptad_raw.sh`
   - `FULL_RUN=1 bash scripts/run_padim_raw.sh`
   - `FULL_RUN=1 bash scripts/run_patchcore_raw.sh`
4. **Stage 2 + 3 — main paper figures** (aggregate CSV, then `outputs/figures/{sec3_promptad,sec3_padim,section4}/`):
   ```bash
   bash scripts/rebuild_main_figures.sh
   ```
   Runs Stage 2 aggregation when needed, then Stage 3 figure builders. Does **not** run Stage 0/1.
5. **Stage 4 — appendix figures** (**partial**):
   ```bash
   bash scripts/rebuild_appendix_figures.sh
   ```
6. Task-to-script detail: `docs/FIGURE_MAP.md`. Legacy fast path: `bash scripts/reproduce_main.sh` (stub Stage 2 only).

## Pipeline summary

| Stage | Entry | Output |
|-------|--------|--------|
| 0 | `scripts/stage0_prepare.sh` | Checks only |
| 1 | `scripts/run_*_raw.sh` | `outputs/cached_results/raw_scores/*/unified_raw_scores_*` |
| 2 | `scripts/run_promptad_pairwise_aggregation.py`, section `analyze_*` | Aggregate CSV under `outputs/cached_results/` |
| 3 | `scripts/rebuild_main_figures.sh` (fig builders) | `outputs/figures/sec3_promptad`, `sec3_padim`, `section4` |
| 4 | `scripts/rebuild_appendix_figures.sh` | `outputs/figures/app_*` (partial) |

Deprecated: `scripts/rebuild_figures_from_raw.sh` → wrapper calling `rebuild_main_figures.sh`.

## Data Policy

This repository intentionally excludes large datasets, model checkpoints, and full intermediate outputs.
Only lightweight CSV, plotting scripts, sample outputs, and reproducibility documentation are tracked.

## Contact

For questions or issues, please open a GitHub issue.
