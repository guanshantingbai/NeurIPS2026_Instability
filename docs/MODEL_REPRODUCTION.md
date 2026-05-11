# Model-level reproduction (full path)

Fast path behavior in `scripts/reproduce_*.sh` and section `run.sh` files is unchanged unless **`FULL_RUN=1`** is set.

This document describes **PatchCore** full-path entrypoints first. PromptAD and PaDiM are out of scope for this phase.

## PatchCore (Appendix F + raw scores)

### Layout

| Stage | Output |
|-------|--------|
| Raw GPU scoring + unified export | `outputs/cached_results/raw_scores/patchcore/` |
| Downstream analyze (pairwise, margin, mechanism, figures) | `outputs/cached_results/app_patchcore_tta/` + copies of PNG/PDF to `outputs/figures/app_patchcore_tta/` |

### Environment variables

| Variable | Required for full run | Meaning |
|----------|----------------------|---------|
| `PATCHCORE_DATA_ROOT` | yes | MVTec root (parent of `bottle/`, …) or VisA root per `patchcore-inspection` loaders |
| `PATCHCORE_MODELS_RUN` | yes | Directory containing `mvtec_<class>/` or `visa_<class>/` PatchCore checkpoints |
| `PATCHCORE_DATASET` | no | `mvtec` (default) or `visa` |
| `PATCHCORE_RAW_OUT` | no | Override raw output directory (default: `outputs/cached_results/raw_scores/patchcore`) |
| `PATCHCORE_GPU` | no | GPU id (default `0`) |
| `PATCHCORE_RESUME` | no | Set to `1` to pass `--resume` on scoring (append classes) |
| `PATCHCORE_FAISS_GPU` | no | Set to `1` to pass `--faiss-on-gpu` |
| `PATCHCORE_INFERENCE_BATCH_SIZE` | no | Batch size (default `24`) |
| `PATCHCORE_EXTRA_ARGS` | no | Space-separated extra CLI tokens for the score step (e.g. `--max-classes 2`) |

### Raw unified score files

After `scripts/run_patchcore_raw.sh`, the directory `outputs/cached_results/raw_scores/patchcore/` contains:

- `patchcore_tta_scores.csv` — upstream per-image scores (identity + TTA views).
- `unified_raw_scores.csv` — one row per image with columns including:
  - `sample_id`, `label`, `identity_score`, `fused_score` (same as identity for PatchCore anomaly score),
  - `condition_scores` / `view_scores` (JSON with `identity`, `horizontal_flip`, `rotate_plus_5deg`, `rotate_minus_5deg`),
  - `dataset`, `category`, `transform`, `config`, `image_path`.
- `unified_raw_scores_long.csv` — long form with `transform` / `condition` per view.

### Commands (acceptance)

**1) Raw scoring + unified export only**

```bash
export PATCHCORE_DATA_ROOT=/path/to/mvtec_parent
export PATCHCORE_MODELS_RUN=/path/to/patchcore_models/models   # example layout
export PATCHCORE_DATASET=mvtec   # optional
# export PATCHCORE_RESUME=1      # if continuing after partial class list

FULL_RUN=1 bash scripts/run_patchcore_raw.sh
```

**2) Full appendix F pipeline (raw + analyze + figures)**

```bash
export PATCHCORE_DATA_ROOT=...
export PATCHCORE_MODELS_RUN=...

FULL_RUN=1 bash scripts/reproduce_app_patchcore_tta.sh
```

`reproduce_app_patchcore_tta.sh` delegates to `src/experiments/app_patchcore_tta/run.sh`, which:

1. Runs `scripts/run_patchcore_raw.sh` (scores + unified tables).
2. Runs `run_patchcore_tta_mechanism.py --step analyze` on `raw_scores/patchcore/patchcore_tta_scores.csv`.
3. Copies generated PNG/PDF into `outputs/figures/app_patchcore_tta/`.

### Notes and limitations

- **First-time scores:** if `patchcore_tta_scores.csv` already exists in the raw output dir, the upstream script exits unless you set **`PATCHCORE_RESUME=1`** or delete the CSV.
- **Hardware:** scoring requires PyTorch + GPU in typical setups; CPU-only may be partial or impractical — mark as **partial** if your environment cannot complete scoring.
- **No edits** to `external/patchcore-inspection` logic beyond what already exists; wrappers live under `scripts/` and `src/experiments/app_patchcore_tta/`.
