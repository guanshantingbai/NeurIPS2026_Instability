# Model-level reproduction (two-stage)

**Stage 1** — `scripts/run_*_raw.sh` (real inference / scoring; each script gates with `FULL_RUN=1` as documented below).

**Stage 2** — `scripts/reproduce_app_patchcore_tta.sh`, `scripts/reproduce_sec3_padim.sh`, `scripts/reproduce_app_padim_representation.sh` and matching `run.sh` files: **consume** existing raw or cached artifacts only; **never** invoke Stage 1. Setting **`FULL_RUN=1` alone does not trigger** `run_*_raw.sh` from `reproduce_*`.

This document describes **PatchCore**, **PaDiM**, and **PromptAD Stage 1** entrypoints (PromptAD Stage 2 section wiring is unchanged in this pass).

- PatchCore: **`docs/FULLPATH_PATCHCORE.md`**
- PaDiM: **`docs/FULLPATH_PADIM.md`**
- PromptAD: **`docs/FULLPATH_PROMPTAD.md`**

## PatchCore (Appendix F + raw scores)

### Layout

| Stage | Script / consumer | Output |
|-------|---------------------|--------|
| **1** | `FULL_RUN=1 bash scripts/run_patchcore_raw.sh` | `outputs/cached_results/raw_scores/patchcore/` (`patchcore_tta_scores.csv`, unified CSVs) |
| **2** | `bash scripts/reproduce_app_patchcore_tta.sh` | If raw unified long exists: analyze → `outputs/cached_results/app_patchcore_tta/` + `outputs/figures/app_patchcore_tta/`; else fast path copy from `result_analysis/patchcore_tta/` |

### Environment variables

| Variable | Required for Stage 1 scoring | Meaning |
|----------|------------------------------|---------|
| `PATCHCORE_DATA_ROOT` | yes (Stage 1) | MVTec root (parent of `bottle/`, …) or VisA root per `patchcore-inspection` loaders |
| `PATCHCORE_MODELS_RUN` | yes (Stage 1) | Directory containing `mvtec_<class>/` or `visa_<class>/` PatchCore checkpoints |
| `PATCHCORE_DATASET` | no | `mvtec` (default) or `visa` |
| `PATCHCORE_RAW_OUT` | no | Override raw output directory (default: `outputs/cached_results/raw_scores/patchcore`) |
| `PATCHCORE_GPU` | no | GPU id (default `0`) |
| `PATCHCORE_RESUME` | no | Set to `1` to pass `--resume` on scoring (append classes) |
| `PATCHCORE_FAISS_GPU` | no | Set to `1` to pass `--faiss-on-gpu` |
| `PATCHCORE_INFERENCE_BATCH_SIZE` | no | Batch size (default `24`) |
| `PATCHCORE_EXTRA_ARGS` | no | Space-separated extra CLI tokens for the score step (e.g. `--max-classes 2`) |
| `PATCHCORE_CLASSES` | no | Comma-separated class subset for the score step (e.g. `bottle` for smoke tests) |

### Raw unified score files

After `scripts/run_patchcore_raw.sh`, the directory `outputs/cached_results/raw_scores/patchcore/` contains:

- `patchcore_tta_scores.csv` — upstream per-image scores (identity + TTA views).
- `unified_raw_scores.csv` — one row per image with columns including:
  - `sample_id`, `label`, `identity_score`, `base_score`, `fused_score` (PatchCore uses the identity view as the primary anomaly score),
  - `condition_scores` / `view_scores` (JSON with `identity`, `horizontal_flip`, `rotate_plus_5deg`, `rotate_minus_5deg`),
  - `dataset`, `category`, `transform`, `condition`, `config`, `image_path`.
- `unified_raw_scores_long.csv` — long form with `transform` / `condition`, `condition_score` (and `score` alias), plus `base_score` / `fused_score` for traceability.

### Commands (acceptance)

**1) Raw scoring + unified export only**

`scripts/run_patchcore_raw.sh` **requires** `FULL_RUN=1` (it will exit otherwise).

```bash
export PATCHCORE_DATA_ROOT=/path/to/mvtec_parent
export PATCHCORE_MODELS_RUN=/path/to/patchcore_models/models   # example layout
export PATCHCORE_DATASET=mvtec   # optional
# export PATCHCORE_RESUME=1      # if continuing after partial class list
# export PATCHCORE_CLASSES=bottle   # optional subset for smoke tests

FULL_RUN=1 bash scripts/run_patchcore_raw.sh
```

**2) Stage 2 — Appendix F from cached raw evidence (analyze only)**

After Stage 1 produced `unified_raw_scores_long.csv` (and `patchcore_tta_scores.csv`):

```bash
export PATCHCORE_DATA_ROOT=...
export PATCHCORE_MODELS_RUN=...
bash scripts/reproduce_app_patchcore_tta.sh
```

`reproduce_app_patchcore_tta.sh` → `src/experiments/app_patchcore_tta/run.sh`:

1. If raw unified long exists: runs `run_patchcore_tta_mechanism.py --step analyze` (no scoring).
2. Copies PNG/PDF into `outputs/figures/app_patchcore_tta/`, mirrors unified CSVs into `outputs/cached_results/app_patchcore_tta/`.
3. If raw evidence is absent: fast path copies from `result_analysis/patchcore_tta/`. Use **`PATCHCORE_FROM_RAW=1`** to require raw files and error if missing.

### Notes and limitations

- **First-time scores:** if `patchcore_tta_scores.csv` already exists in the raw output dir, the upstream script exits unless you set **`PATCHCORE_RESUME=1`** or delete the CSV.
- **Hardware:** scoring requires PyTorch + GPU in typical setups; CPU-only may be partial or impractical — mark as **partial** if your environment cannot complete scoring.
- **No edits** to `external/patchcore-inspection` logic beyond what already exists; wrappers live under `scripts/` and `src/experiments/app_patchcore_tta/`.
- **Blocker / partial:** if `PATCHCORE_DATA_ROOT` or `PATCHCORE_MODELS_RUN` is wrong, scoring fails immediately (e.g. missing `mvtec_<class>/patchcore_params.pkl`). See **`docs/FULLPATH_PATCHCORE.md`** for the maintainer verification note.

---

## PaDiM (Section 3.1.2 + Appendix E, Protocol B raw)

### Layout

| Stage | Script / consumer | Output |
|-------|---------------------|--------|
| **1** | `FULL_RUN=1 bash scripts/run_padim_raw.sh` | Jobs under `PADIM_OUTPUT_ROOT/protocol_b_jobs/…`, then aggregated `outputs/cached_results/raw_scores/padim/`, `marginal_protocol_b.csv`, `mechanism_from_raw.csv` |
| **2** | `bash scripts/reproduce_sec3_padim.sh` / `bash scripts/reproduce_app_padim_representation.sh` | Plots or copies **only** from existing `marginal_protocol_b.csv`, `mechanism_from_raw.csv`, or bundled stubs — **no** `run_padim_raw.sh` |

### Environment variables

| Variable | Required for `run_padim_raw.sh` | Meaning |
|----------|----------------------------------|---------|
| `FULL_RUN` | must be `1` | Gate for real inference |
| `PADIM_DATA_ROOT` | yes | Dataset root passed to `padim_protocol_b_one_run.py --data_path` |
| `PADIM_OUTPUT_ROOT` | yes | Writable root; jobs live under `.../protocol_b_jobs/` |
| `PADIM_CLASSES` | no\* | Comma-separated class names. **Optional** when `PADIM_PROFILE` auto-fill applies (`mvtec` + unset classes); otherwise **required**. Explicit list **always** overrides profile. |
| `PADIM_PROFILE` | no | `debug` \| `paper` (default **`paper`**). When `PADIM_CLASSES` is unset: `paper` + `mvtec` → MVTec-15 list; `debug` → five-class smoke list. Non-`mvtec`: set `PADIM_CLASSES`. See **`docs/FULLPATH_PADIM.md`**. |
| `PADIM_BACKBONES` | yes | Comma-separated `resnet18` and/or `wide_resnet50_2` |
| `PADIM_SEEDS` | yes | Comma-separated integer seeds |
| `PADIM_DATASET` | no | `mvtec` (default) or `visa` |
| `PADIM_GPU` | no | Sets `CUDA_VISIBLE_DEVICES` (default `0`) |
| `PADIM_FORCE` | no | `1` re-runs jobs even if `per_sample.csv` exists |
| `PADIM_EXTRA_ARGS` | no | Space-separated extra CLI tokens (e.g. `--cov-float32 --max-train-images 350`) |

\*`PADIM_CLASSES` may be omitted only when **`PADIM_PROFILE`** supplies defaults (`mvtec` dataset). For **`visa`** or custom layouts, set **`PADIM_CLASSES`** explicitly.

```bash
export PADIM_DATA_ROOT=/path/to/mvtec_or_visa
export PADIM_OUTPUT_ROOT=/path/to/scratch_or_outputs/padim_runs
export PADIM_PROFILE=paper   # default; omit PADIM_CLASSES for canonical MVTec-15 on mvtec
export PADIM_BACKBONES=resnet18
export PADIM_SEEDS=111,222,333,444,555

FULL_RUN=1 bash scripts/run_padim_raw.sh
bash scripts/reproduce_sec3_padim.sh
bash scripts/reproduce_app_padim_representation.sh
```

Appendix E Stage 2 consumes **`mechanism_from_raw.csv`** when present (produced at end of Stage 1 aggregation). That file is a **raw-score-level partial** summary (cross-seed |ΔAUROC| stats from the **configured** `PADIM_SEEDS`); it is **not** full appendix reproduction and **not** the **`padim_seed_killer_evidence_pipeline.py`** one-click chain; see **`docs/FULLPATH_PADIM.md`**.

### Raw score columns (unified long)

Includes at least: `sample_id`, `label`, `fused_score`, `view_id`, `condition`, `condition_score`, `dataset`, `category`, `backbone`, `seed`, `config`, `image_path`.

### Notes

- Default **fast paths** for sec3 / appendix E are unchanged (bundled stubs under `samples/fastpath/`).
- The legacy **`run_padim_seed_killer_one_click.sh`** wide multi-setting sweep is **not** invoked from Stage 2 `run.sh`; Stage 1 scope is limited by the effective **`PADIM_CLASSES`** (explicit or profile-filled) / `PADIM_SEEDS` / `PADIM_BACKBONES`.

---

## PromptAD (Stage 1 raw evidence — Section 3.1.1 / 4 / Appendix C/G **not** refactored here)

| Stage | Script | Output |
|-------|--------|--------|
| **1** | `FULL_RUN=1 bash scripts/run_promptad_raw.sh` | `outputs/cached_results/raw_scores/promptad/` (`unified_raw_scores_wide.csv`, `unified_raw_scores_long.csv`, `manifest.json`) from existing `CLS-*-per_sample.csv` (**export** mode) and/or optional **train**/**infer** loops |
| **2** | existing `reproduce_*` / section `run.sh` | Unchanged in this pass — still fast-path stubs unless separately wired to consume raw scores later |

Details: **`docs/FULLPATH_PROMPTAD.md`**. Adapter: `python -m src.models.promptad_adapter.run_promptad …`; exporter: `src/models/promptad_adapter/promptad_export_unified_raw.py`.

**Section 3.1.1 (Stage 2):** `src/experiments/sec3_promptad_observation/run.sh` consumes `outputs/cached_results/raw_scores/promptad/unified_raw_scores_{wide,long}.csv` via `analyze_sec3_promptad_from_raw.py` (no PromptAD train/infer). Optional stub: `SEC3_PROMPTAD_ALLOW_STUB=1`.
