# Figure/Table Map (fast path vs full run)

Canonical pipeline: **`docs/PIPELINE_STAGES.md`** (Stages 0–4). Legacy `reproduce_*.sh` fast paths map to Stage 2/4 stubs without model scoring. **Stage 1** = `run_*_raw.sh` — see `docs/REPRODUCE.md`.

Full paper figures still require real data / Stage 1 artifacts where applicable. See `docs/REPRODUCIBILITY_STATUS.md`.

## Main paper (fast path outputs — markers / stubs only)

| Paper ref | Section | Script | Fast path outputs (actual paths) |
|-----------|---------|--------|----------------------------------|
| Fig 1 (observation) | 3.1.1 PromptAD | `src/experiments/sec3_promptad_observation/run.sh` | **From raw (default):** consumes `outputs/cached_results/raw_scores/promptad/unified_raw_scores_{wide,long}.csv` → tables + diagnostic PNGs under `outputs/cached_results/sec3_promptad/` and `outputs/figures/sec3_promptad/` (see `analyze_sec3_promptad_from_raw.py`). **Paper-style Fig 1 (cached only, no pairwise recompute):** `python3 src/experiments/sec3_promptad_observation/build_paper_style_fig1.py` → `outputs/figures/sec3_promptad/paper_style_fig1.png`, `paper_style_fig1.pdf`, and `outputs/cached_results/sec3_promptad/paper_style_fig1_summary.json`. **Stub:** `SEC3_PROMPTAD_ALLOW_STUB=1` restores bundled CSV + marker only. Stage 1 export: **`docs/FULLPATH_PROMPTAD.md`**. |
| Fig 2 (observation) | 3.1.2 PaDiM | `src/experiments/sec3_padim_observation/run.sh` | **Fast path:** `outputs/cached_results/sec3_padim/marginal_stub.csv`, `outputs/figures/sec3_padim/fastpath_done.txt`. **Stage 1:** `FULL_RUN=1 bash scripts/run_padim_raw.sh` with **`PADIM_PROFILE`** (`debug` = smoke five-class auto-fill; **`paper`** = default, canonical **MVTec-15** when `PADIM_CLASSES` unset on `mvtec`) — see **`docs/FULLPATH_PADIM.md`**. **Full path outputs:** `outputs/cached_results/raw_scores/padim/`, `marginal_protocol_b.csv`, optional diagnostic `padim_marginal_scatter.png`. **Paper-style evidence 2×2 (cached marginal + wide raw, no PaDiM rerun):** `python3 src/experiments/sec3_padim_observation/build_paper_style_fig2_padim.py` → `outputs/figures/sec3_padim/paper_style_fig2.{png,pdf}` + `outputs/cached_results/sec3_padim/paper_style_fig2_summary.json`. |
| Fig 3–6 (diagnostics) | 4 Systematic | `src/experiments/sec4_systematic_validation/run.sh` | **Stage 2 (optional legacy):** `analyze_sec4_promptad_from_raw.py` → diagnostic PNGs under `outputs/figures/sec4_systematic/` (not main paper). **Stage 3 paper bundle:** `build_section4_paper_figures.py` → `outputs/figures/section4/{fig2_same_auroc,fig3_4_merged_mechanism,fig5_6_merged_failure,fig7_delta_risk}.{png,pdf}`. Orchestrated by **`scripts/rebuild_main_figures.sh`** (Stage 2+3). |

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

- **Main paper figures** (`outputs/figures/sec3_promptad`, `sec3_padim`, `section4`): **`scripts/rebuild_main_figures.sh`** (Stage 2 aggregation + Stage 3 figures; no Stage 0/1). Deprecated alias: `rebuild_figures_from_raw.sh`.
- **Appendix figures** (`outputs/figures/app_*`): **`scripts/rebuild_appendix_figures.sh`** (Stage 4, **partial**).
- Tables / stubs only: `scripts/reproduce_main.sh` (fast path by default)
- Appendix: `scripts/reproduce_appendix.sh` (fast path only by default)
