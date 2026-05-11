# NeurIPS2026_Instability

Paper-section-oriented reproducibility repository for the NeurIPS 2026 instability paper.

## Design Principle

This repository is organized by paper sections and appendix tasks, not by model families.
Each section has explicit inputs, outputs, scripts, and commands.

## Repository Layout

- `docs/`: reproducibility guide, figure/table map, dataset and environment docs.
- `scripts/`: top-level orchestration scripts for main paper and appendix runs.
- `src/core/`: shared analysis logic (pairwise, instability, risk-coverage, metrics).
- `src/experiments/`: section-level experiment packages.
- `src/models/*_adapter/`: thin adapters to external model repositories.
- `external/`: third-party model code (PromptAD, PaDiM, PatchCore).
- `outputs/`: generated figures/tables/cached lightweight results.
- `figures/`: curated figures grouped by main paper vs appendix.

## Quick Start

1. Read `docs/ENVIRONMENT.md` and `docs/DATASET.md`.
2. Check task-to-script mapping in `docs/FIGURE_MAP.md` (fast path vs raw-derived Stage 2).
3. **Two-stage layout** (see `docs/REPRODUCE.md`):
   - **Stage 1 — model evidence:** `scripts/run_patchcore_raw.sh`, `scripts/run_padim_raw.sh` (each requires `FULL_RUN=1` plus data/env; **not** invoked by `reproduce_*`).
   - **Stage 2 — section reproduction:** `scripts/reproduce_app_patchcore_tta.sh`, `scripts/reproduce_sec3_padim.sh`, `scripts/reproduce_app_padim_representation.sh` (consume existing raw/cached assets only; **no** scoring).
4. Run the **default Stage 2 fast path** (bundled stubs + cached PatchCore TTA assets where raw scores are absent):
   - `bash scripts/reproduce_main.sh`
   - `bash scripts/reproduce_appendix.sh`
5. End-to-end paper reruns: run Stage 1 with real data, then Stage 2; see `docs/REPRODUCIBILITY_STATUS.md`.

## Current Scope

- Section-level `reproduce_*` scripts are **Stage 2 only** (cached / raw-derived inputs; exit non-zero if required inputs are missing).
- **`FULL_RUN=1`** on **`run_*_raw.sh`** runs Stage 1 model extraction for PatchCore / PaDiM; **`reproduce_*` does not auto-call those scripts**.

## Data Policy

This repository intentionally excludes large datasets, model checkpoints, and full intermediate outputs.
Only lightweight CSV, plotting scripts, sample outputs, and reproducibility documentation are tracked.

## Contact

For questions or issues, please open a GitHub issue.
