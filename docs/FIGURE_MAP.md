# Figure/Table Map (fast path vs full run)

Default `scripts/reproduce_*.sh` runs **Stage 2 fast path**: bundled CSV stubs, cached PatchCore TTA assets, and **no** model scoring. **Stage 1** (`run_*_raw.sh` for PatchCore / PaDiM / PromptAD) is separate — see `docs/REPRODUCE.md`.

Full paper figures still require real data / Stage 1 artifacts where applicable. See `docs/REPRODUCIBILITY_STATUS.md`.

## Main paper (fast path outputs — markers / stubs only)

| Paper ref | Section | Script | Fast path outputs (actual paths) |
|-----------|---------|--------|----------------------------------|
| Fig 1 (observation) | 3.1.1 PromptAD | `src/experiments/sec3_promptad_observation/run.sh` | **From raw (default):** consumes `outputs/cached_results/raw_scores/promptad/unified_raw_scores_{wide,long}.csv` → tables + PNGs under `outputs/cached_results/sec3_promptad/` and `outputs/figures/sec3_promptad/` (see `analyze_sec3_promptad_from_raw.py`). **Stub:** `SEC3_PROMPTAD_ALLOW_STUB=1` restores bundled CSV + marker only. Stage 1 export: **`docs/FULLPATH_PROMPTAD.md`**. |
| Fig 2 (observation) | 3.1.2 PaDiM | `src/experiments/sec3_padim_observation/run.sh` | **Fast path:** `outputs/cached_results/sec3_padim/marginal_stub.csv`, `outputs/figures/sec3_padim/fastpath_done.txt`. **Full path (`FULL_RUN=1` + `PADIM_*`):** `outputs/cached_results/raw_scores/padim/`, `marginal_protocol_b.csv`, optional `padim_marginal_scatter.png` — see **`docs/FULLPATH_PADIM.md`**. |
| Fig 3–6 | 4 Systematic | `src/experiments/sec4_systematic_validation/run.sh` | **From PromptAD unified raw (optional):** if `outputs/cached_results/raw_scores/promptad/unified_raw_scores_{wide,long}.csv` exists → `analyze_sec4_promptad_from_raw.py` writes pairwise-based CSVs under `outputs/cached_results/sec4_systematic/` and PNGs (`fig_controlled_margin.png`, `fig_failure_conditioned_signals.png`, `fig_instability_regime_failure_rate.png`, `fig_decision_consequence_delta_risk.png`) + `from_promptad_raw_done.txt` — **no** `external/PromptAD` scripts. **Stub fast path (unchanged):** if raw is missing, `fastpath_done.txt` under `outputs/figures/sec4_systematic/` and `outputs/tables/sec4_systematic/`; local `compute_*.py` placeholders only. |

**Minimal / medium full-path (not paper 81-setting grid):** **`docs/FULLPATH_PROMPTAD.md`** **(C)** (two seeds), **(D)** (five seeds: `111–555`, `outputs/promptad_fullpath_bottle_k1_5seeds/`), and **(F)** (27-setting `bottle,cable,capsule` × `1,2,4` × three seeds + **`docs/FULLPATH_PROMPTAD.md`** **(E)** resume/status) document on-host **`minimal` / `small-scale` / `medium-scale PromptAD full-path verified`** runs — see **`docs/END_TO_END_VALIDATION.md`** §11 / §12 / §13.

**Full-run-required:** Camera-ready paper Fig 3–6 matching the original PromptAD strengthening pipeline still needs `FULL_RUN=1` and `external/PromptAD` experiments plus pilot CSVs — **not** the unified-raw branch above (that branch is a **pairwise reproduction from Stage 1 export scores** only).

## Appendix (fast path)

| Appendix | Script | Fast path outputs |
|----------|--------|-------------------|
| C | `src/experiments/app_promptad_generalization/run.sh` | **Fast path:** `spearman_stub.csv` + marker. Future raw-join may consume **`outputs/cached_results/raw_scores/promptad/`** from Stage 1 — **`docs/FULLPATH_PROMPTAD.md`**. |
| E | `src/experiments/app_padim_representation/run.sh` (via `scripts/reproduce_app_padim_representation.sh`) | **Fast path:** `mechanism_stub.csv` + `fastpath_done.txt`. **Stage 2 with Stage 1 outputs:** `mechanism_from_raw.csv` (+ `fullrun_done.txt`) after **`run_padim_raw.sh`** — raw-score-level **partial** only; **not** full seed-killer pipeline — **`docs/FULLPATH_PADIM.md`**. |
| F | `src/experiments/app_patchcore_tta/run.sh` (via `scripts/reproduce_app_patchcore_tta.sh`) | **Stage 2:** if `outputs/cached_results/raw_scores/patchcore/unified_raw_scores_long.csv` exists → **analyze only** from cached raw (Stage 1: `run_patchcore_raw.sh`). Else **fast path:** copy from `result_analysis/patchcore_tta/`. `reproduce_*` never runs Stage 1. See **`docs/REPRODUCE.md`**, **`docs/FULLPATH_PATCHCORE.md`**. |
| G | `src/experiments/app_signal_comparison/run.sh` | **Fast path:** `outputs/figures/app_signal_comparison/*.csv`, `*.png`, `*.pdf`; mirrored CSV under `outputs/tables/app_signal_comparison/`. Optional future alignment with PromptAD unified raw — **`docs/FULLPATH_PROMPTAD.md`**. |

## Orchestration

- Main paper: `scripts/reproduce_main.sh` (fast path only by default)
- Appendix: `scripts/reproduce_appendix.sh` (fast path only by default)
