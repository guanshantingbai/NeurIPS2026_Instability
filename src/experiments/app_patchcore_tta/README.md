# Appendix F PatchCore + TTA

Reproduces input-level equivalent-condition analyses:

- TTA construction;
- controlled margin analysis;
- margin -> instability -> error chain.

**Stage 1 (model evidence):** `FULL_RUN=1 bash scripts/run_patchcore_raw.sh` with `PATCHCORE_*` — writes `outputs/cached_results/raw_scores/patchcore/` (see `docs/FULLPATH_PATCHCORE.md`).

**Stage 2 (this `run.sh` via `scripts/reproduce_app_patchcore_tta.sh`):** never runs Stage 1. If `unified_raw_scores_long.csv` exists under raw scores, runs **analyze only**; else copies from `result_analysis/patchcore_tta/`. Use **`PATCHCORE_FROM_RAW=1`** to require raw files. See **`docs/REPRODUCE.md`**.
