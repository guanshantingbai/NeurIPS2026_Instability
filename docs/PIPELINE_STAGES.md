# Pipeline stages (canonical)

This repository uses a **five-stage** layout. Older docs that say “Stage 1 / Stage 2 only” map onto Stages **0–4** below.

| Stage | Name | Runs models? | Primary outputs |
|-------|------|--------------|-----------------|
| **0** | Preparation | No | Checks, downloads, sanity |
| **1** | Model evidence | Yes | `unified_raw_scores_*` per model |
| **2** | Pairwise / statistical aggregation | No | Aggregate CSV (+ optional `pairwise_metrics.csv`) |
| **3** | Main-paper figures | No | `outputs/figures/{sec3_promptad,sec3_padim,section4}/` |
| **4** | Appendix figures | No | `outputs/figures/app_*/` (**partial**) |

---

## Stage 0 — Preparation

**Not** model scoring. Ensures the repo and runtime are ready before Stage 1.

Includes (among others):

- Dataset path checks (`PATCHCORE_DATA_ROOT`, `PADIM_DATA_ROOT`, PromptAD dataset roots, etc.)
- `external/` repository presence / submodule checks
- **PatchCore pretrained weights** — verify or download (one sub-step of Stage 0, not a separate pipeline)
- Shell / Python sanity: `bash -n scripts/*.sh`, `python3 -m compileall -q src`

**Planned entry:** `bash scripts/stage0_prepare.sh` (interface; extend as checks are implemented).

---

## Stage 1 — Model evidence generation

Produces **unified raw score tables** from real upstream runs or export-from-existing CSVs.

| Model | Script | Typical outputs |
|-------|--------|-----------------|
| PromptAD | `FULL_RUN=1 bash scripts/run_promptad_raw.sh` | `outputs/cached_results/raw_scores/promptad/unified_raw_scores_{wide,long}.csv` |
| PaDiM | `FULL_RUN=1 bash scripts/run_padim_raw.sh` | `outputs/cached_results/raw_scores/padim/`, `marginal_protocol_b.csv`, … |
| PatchCore | `FULL_RUN=1 bash scripts/run_patchcore_raw.sh` | `outputs/cached_results/raw_scores/patchcore/unified_raw_scores_long.csv`, … |

See `docs/FULLPATH_PROMPTAD.md`, `docs/FULLPATH_PADIM.md`, `docs/FULLPATH_PATCHCORE.md`.

**Does not include:** pairwise aggregation, near-AUROC universe, or paper figure rendering.

---

## Stage 2 — Pairwise / statistical aggregation

Reads Stage 1 **unified raw** (no train/infer). Writes **tables only** — no final main-paper figure PNG/PDF bundle.

Examples:

- `scripts/run_promptad_pairwise_aggregation.py` — vectorized pairwise; `setting_level_metrics.csv`, optional `pairwise_metrics.csv`, controlled margin tables
- `src/experiments/sec3_promptad_observation/analyze_sec3_promptad_from_raw.py` — `risk_coverage.csv`, `same_auroc_instability_pairs.csv`, `sample_level_metrics.csv` (may also emit diagnostic plots as a side effect; those are not Stage 3 paper figures)
- PaDiM / PatchCore section analyzers under `src/experiments/*/run.sh` when driven from raw

Caches live under `outputs/cached_results/{sec3_promptad,sec4_systematic,promptad_pairwise,...}/`.

---

## Stage 3 — Main-paper figure generation

Reads **Stage 2 aggregate CSV/JSON** only (no unified raw recompute in the paper-style builders).

| Output dir | Builder |
|------------|---------|
| `outputs/figures/sec3_promptad/` | `build_paper_style_fig1.py` (+ optional diagnostics from Stage 2) |
| `outputs/figures/sec3_padim/` | `build_paper_style_fig2_padim.py` (if present), `plot_padim_observation.py` |
| `outputs/figures/section4/` | `build_section4_paper_figures.py` |

**Convenience entry (Stage 2 + 3):** `bash scripts/rebuild_main_figures.sh`  
Runs Stage 2 aggregation when caches are missing or `FORCE_REBUILD=1`, then Stage 3 figure builders. **Does not run Stage 0 or Stage 1.**

Legacy name: `scripts/rebuild_figures_from_raw.sh` → deprecated wrapper.

---

## Stage 4 — Appendix figure generation

Appendix panels under `outputs/figures/app_*` (PatchCore TTA, PaDiM representation, PromptAD generalization, signal comparison, …).

**Status: partial.** Interface only:

```bash
bash scripts/rebuild_appendix_figures.sh
```

Delegates to `scripts/reproduce_appendix.sh` (Stage 2 fast path / cached assets) until a dedicated Stage 4 bundle exists.

---

## Quick command map

```bash
# Stage 0 (checks — extend over time)
bash scripts/stage0_prepare.sh

# Stage 1 (GPU / data — per model)
FULL_RUN=1 bash scripts/run_promptad_raw.sh
FULL_RUN=1 bash scripts/run_padim_raw.sh
FULL_RUN=1 bash scripts/run_patchcore_raw.sh

# Stage 2 only (example: PromptAD pairwise)
PROMPTAD_SAVE_PAIRWISE=1 bash scripts/run_promptad_pairwise_aggregation.py --workers 8 --save-pairwise

# Stage 2 + 3 main paper figures
bash scripts/rebuild_main_figures.sh

# Stage 4 appendix (partial)
bash scripts/rebuild_appendix_figures.sh
```

See also `docs/REPRODUCE.md`, `docs/FIGURE_MAP.md`.
