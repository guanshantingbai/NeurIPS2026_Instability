# Figure/Table Map (fast path vs full run)

Default `scripts/reproduce_*.sh` runs the **fast path**: bundled CSV stubs, cached PatchCore TTA assets, and **no** full GPU multi-seed training.

Full paper figures require **`FULL_RUN=1`** on the relevant section (and data / prior artifacts). See `docs/REPRODUCIBILITY_STATUS.md`.

## Main paper (fast path outputs — markers / stubs only)

| Paper ref | Section | Script | Fast path outputs (actual paths) |
|-----------|---------|--------|----------------------------------|
| Fig 1 (observation) | 3.1.1 PromptAD | `src/experiments/sec3_promptad_observation/run.sh` | `outputs/cached_results/sec3_promptad/pairwise_stub.csv`, `outputs/figures/sec3_promptad/fastpath_done.txt` |
| Fig 2 (observation) | 3.1.2 PaDiM | `src/experiments/sec3_padim_observation/run.sh` | **Fast path:** `outputs/cached_results/sec3_padim/marginal_stub.csv`, `outputs/figures/sec3_padim/fastpath_done.txt`. **Full path (`FULL_RUN=1` + `PADIM_*`):** `outputs/cached_results/raw_scores/padim/`, `marginal_protocol_b.csv`, optional `padim_marginal_scatter.png` — see **`docs/FULLPATH_PADIM.md`**. |
| Fig 3–6 | 4 Systematic | `src/experiments/sec4_systematic_validation/run.sh` | `outputs/figures/sec4_systematic/fastpath_done.txt`, `outputs/tables/sec4_systematic/fastpath_done.txt` (no real Fig 3–6 PNG/PDF here) |

**Full-run-required:** Regenerating actual Fig 1–6 as in the PDF needs `FULL_RUN=1` and external pipelines (PromptAD / PaDiM / strengthening scripts) plus local datasets and long runs — **not** the default.

## Appendix (fast path)

| Appendix | Script | Fast path outputs |
|----------|--------|-------------------|
| C | `src/experiments/app_promptad_generalization/run.sh` | `outputs/tables/app_promptad_generalization/spearman_stub.csv`, `outputs/figures/app_promptad_generalization/fastpath_done.txt` |
| E | `src/experiments/app_padim_representation/run.sh` (via `scripts/reproduce_app_padim_representation.sh`) | **Fast path:** `mechanism_stub.csv` + `fastpath_done.txt`. **Stage 2 with Stage 1 outputs:** `mechanism_from_raw.csv` (+ `fullrun_done.txt`) after **`run_padim_raw.sh`** — raw-score-level **partial** only; **not** full seed-killer pipeline — **`docs/FULLPATH_PADIM.md`**. |
| F | `src/experiments/app_patchcore_tta/run.sh` (via `scripts/reproduce_app_patchcore_tta.sh`) | **Stage 2:** if `outputs/cached_results/raw_scores/patchcore/unified_raw_scores_long.csv` exists → **analyze only** from cached raw (Stage 1: `run_patchcore_raw.sh`). Else **fast path:** copy from `result_analysis/patchcore_tta/`. `reproduce_*` never runs Stage 1. See **`docs/REPRODUCE.md`**, **`docs/FULLPATH_PATCHCORE.md`**. |
| G | `src/experiments/app_signal_comparison/run.sh` | `outputs/figures/app_signal_comparison/*.csv`, `*.png`, `*.pdf`; mirrored CSV under `outputs/tables/app_signal_comparison/` |

## Orchestration

- Main paper: `scripts/reproduce_main.sh` (fast path only by default)
- Appendix: `scripts/reproduce_appendix.sh` (fast path only by default)
