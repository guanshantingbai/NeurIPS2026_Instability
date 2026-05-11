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
2. Check task-to-script mapping in `docs/FIGURE_MAP.md` (fast path vs full run).
3. Run the **default fast path** (bundled stubs + cached PatchCore TTA assets; no full GPU reruns):
   - `bash scripts/reproduce_main.sh`
   - `bash scripts/reproduce_appendix.sh`
4. Full paper reruns require `FULL_RUN=1` and external data; see `docs/REPRODUCIBILITY_STATUS.md`.

## Current Scope

- Section-level scripts are **honest fast paths** by default (exit non-zero if required cached inputs are missing).
- **`FULL_RUN=1`** enables heavy external pipelines (PromptAD / PaDiM / PatchCore / strengthening); not validated in CI by default.

## Data Policy

This repository intentionally excludes large datasets, model checkpoints, and full intermediate outputs.
Only lightweight CSV, plotting scripts, sample outputs, and reproducibility documentation are tracked.

## Contact

For questions or issues, please open a GitHub issue.
