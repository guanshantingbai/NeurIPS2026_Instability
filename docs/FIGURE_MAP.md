# Figure/Table Map (fast path vs full run)

Default `scripts/reproduce_*.sh` runs the **fast path**: bundled CSV stubs, cached PatchCore TTA assets, and **no** full GPU multi-seed training.

Full paper figures require **`FULL_RUN=1`** on the relevant section (and data / prior artifacts). See `docs/REPRODUCIBILITY_STATUS.md`.

## Main paper (fast path outputs — markers / stubs only)

| Paper ref | Section | Script | Fast path outputs (actual paths) |
|-----------|---------|--------|----------------------------------|
| Fig 1 (observation) | 3.1.1 PromptAD | `src/experiments/sec3_promptad_observation/run.sh` | `outputs/cached_results/sec3_promptad/pairwise_stub.csv`, `outputs/figures/sec3_promptad/fastpath_done.txt` |
| Fig 2 (observation) | 3.1.2 PaDiM | `src/experiments/sec3_padim_observation/run.sh` | `outputs/cached_results/sec3_padim/marginal_stub.csv`, `outputs/figures/sec3_padim/fastpath_done.txt` |
| Fig 3–6 | 4 Systematic | `src/experiments/sec4_systematic_validation/run.sh` | `outputs/figures/sec4_systematic/fastpath_done.txt`, `outputs/tables/sec4_systematic/fastpath_done.txt` (no real Fig 3–6 PNG/PDF here) |

**Full-run-required:** Regenerating actual Fig 1–6 as in the PDF needs `FULL_RUN=1` and external pipelines (PromptAD / PaDiM / strengthening scripts) plus local datasets and long runs — **not** the default.

## Appendix (fast path)

| Appendix | Script | Fast path outputs |
|----------|--------|-------------------|
| C | `src/experiments/app_promptad_generalization/run.sh` | `outputs/tables/app_promptad_generalization/spearman_stub.csv`, `outputs/figures/app_promptad_generalization/fastpath_done.txt` |
| E | `src/experiments/app_padim_representation/run.sh` | `outputs/tables/app_padim_representation/mechanism_stub.csv`, `outputs/figures/app_padim_representation/fastpath_done.txt` |
| F | `src/experiments/app_patchcore_tta/run.sh` | `outputs/cached_results/app_patchcore_tta/*.csv`, `outputs/figures/app_patchcore_tta/*.png`, `*.pdf` (copied from `result_analysis/patchcore_tta/`) |
| G | `src/experiments/app_signal_comparison/run.sh` | `outputs/figures/app_signal_comparison/*.csv`, `*.png`, `*.pdf`; mirrored CSV under `outputs/tables/app_signal_comparison/` |

## Orchestration

- Main paper: `scripts/reproduce_main.sh` (fast path only by default)
- Appendix: `scripts/reproduce_appendix.sh` (fast path only by default)
