# PatchCore full-path reproduction (Appendix F)

This document covers **only** the PatchCore + TTA mechanism pipeline: regenerate **raw scores**, **cached downstream CSVs**, and **Appendix F figures** from real `external/patchcore-inspection` inference. PromptAD and PaDiM are out of scope.

**Two-stage alignment:** `scripts/run_patchcore_raw.sh` is **Stage 1** (model evidence). `scripts/reproduce_app_patchcore_tta.sh` is **Stage 2** only: it **never** calls Stage 1; with existing `unified_raw_scores_long.csv` it runs **analyze**; otherwise it uses the fast path copy from `result_analysis/patchcore_tta/`.

---

## Environment requirements

- **Python 3** with PyTorch, torchvision, pandas, numpy, matplotlib, Pillow, tqdm, and a working **FAISS** build compatible with `patchcore-inspection` (CPU FAISS is enough; optional `faiss-gpu` for `--faiss-on-gpu`).
- **CUDA GPU** strongly recommended for scoring all MVTec/VisA classes in reasonable time; analyze-only is CPU-bound.
- Repository root on **`PYTHONPATH`** (handled automatically by sourcing `scripts/repo_env.sh` from the wrapper scripts).

---

## Data preparation

1. **Images** — set `PATCHCORE_DATA_ROOT` to the dataset root expected by upstream loaders:
   - **MVTec AD:** parent directory that contains class folders (`bottle/`, `cable/`, …), each with `test/good/` and `test/<defect>/`.
   - **VisA:** root layout as in `patchcore-inspection` `VisADataset` (see upstream `patchcore/datasets/visa.py`).

2. **PatchCore checkpoints** — set `PATCHCORE_MODELS_RUN` to a directory containing one subdirectory per class, named:
   - `mvtec_<classname>/` for MVTec, or  
   - `visa_<classname>/` for VisA,  
   each with `patchcore_params.pkl` and the saved PatchCore artifacts produced by the upstream training/export flow (see `external/patchcore-inspection` README and `scripts/download_mvtec_patchcore_lfs.sh` for reference layouts).

---

## Running commands

From the repository root:

### A) Raw scores + unified tables only

Requires **`FULL_RUN=1`** (the raw script refuses to run otherwise).

```bash
export PATCHCORE_DATA_ROOT=/path/to/mvtec_or_visa_root
export PATCHCORE_MODELS_RUN=/path/to/models_parent   # contains mvtec_* / visa_*
export PATCHCORE_DATASET=mvtec   # or visa

# Optional:
# export PATCHCORE_RAW_OUT="$PWD/outputs/cached_results/raw_scores/patchcore"
# export PATCHCORE_GPU=0
# export PATCHCORE_RESUME=1          # if patchcore_tta_scores.csv already exists and you want to append classes
# export PATCHCORE_CLASSES=bottle,cable   # subset for smoke tests
# export PATCHCORE_FAISS_GPU=1
# export PATCHCORE_INFERENCE_BATCH_SIZE=48

FULL_RUN=1 bash scripts/run_patchcore_raw.sh
```

### B) Two-stage Appendix F (Stage 1 scoring, then Stage 2 analyze)

```bash
export PATCHCORE_DATA_ROOT=/path/to/mvtec_or_visa_root
export PATCHCORE_MODELS_RUN=/path/to/models_parent

FULL_RUN=1 bash scripts/run_patchcore_raw.sh
bash scripts/reproduce_app_patchcore_tta.sh
```

`reproduce_app_patchcore_tta.sh` → `run.sh`: **does not** call Stage 1. With `unified_raw_scores_long.csv` present, it runs **analyze only**, copies PNG/PDF, mirrors unified CSVs into `outputs/cached_results/app_patchcore_tta/`.

---

## Output paths

| Artifact | Path |
|----------|------|
| Per-image TTA scores (upstream CSV) | `outputs/cached_results/raw_scores/patchcore/patchcore_tta_scores.csv` |
| Unified wide / long raw tables | `outputs/cached_results/raw_scores/patchcore/unified_raw_scores.csv`, `unified_raw_scores_long.csv` |
| Pairwise + margin + mechanism CSVs | `outputs/cached_results/app_patchcore_tta/patchcore_pairwise_analysis.csv`, `controlled_margin_analysis.csv`, `mechanism_chain_summary.csv` |
| Appendix F figures | `outputs/cached_results/app_patchcore_tta/fig_patchcore_*.png` (+ PDF), copies under `outputs/figures/app_patchcore_tta/` |
| Unified table mirror (full run only) | `outputs/cached_results/app_patchcore_tta/unified_raw_scores*.csv` |

**Git:** `outputs/` is gitignored by default — do not commit large run artifacts.

---

## Raw score schema (unified exports)

Wide file `unified_raw_scores.csv` includes at least: `sample_id`, `label`, `base_score`, `fused_score` (PatchCore identity anomaly score), `condition_scores` / `view_scores` (JSON per-view scores), `dataset`, `category`, `transform`, `condition`, `config`, `image_path`.

Long file `unified_raw_scores_long.csv` includes at least: `sample_id`, `label`, `dataset`, `category`, `transform`, `condition`, `condition_score`, `base_score`, `fused_score`, `config`, `image_path` (plus `score` as an alias of `condition_score`).

---

## GPU and runtime expectations

- **GPU:** one CUDA device (default index `0`, override with `PATCHCORE_GPU`). Scoring loads one PatchCore model per class sequentially.
- **Time:** highly dependent on dataset size, image resolution in `patchcore_params.pkl`, batch size, and disk I/O. Order-of-magnitude guide only: **tens of minutes to several hours** for full MVTec test splits with default batching on a mid-range GPU; use `PATCHCORE_CLASSES` for a single-class smoke run.
- **Resume:** use `PATCHCORE_RESUME=1` if scoring was interrupted; otherwise delete `patchcore_tta_scores.csv` or pass a fresh `PATCHCORE_RAW_OUT`.

---

## Verification status in this repository (honest)

End-to-end GPU scoring was **not** re-run in the maintainer environment (no bundled MVTec/VisA trees or PatchCore checkpoints). A smoke invocation with empty model roots fails predictably with `FileNotFoundError` on `patchcore_params.pkl` under `PATCHCORE_MODELS_RUN/mvtec_<class>/`, confirming the adapter invokes upstream code without a fast-path stub.

With valid `PATCHCORE_DATA_ROOT` and `PATCHCORE_MODELS_RUN`, the intended chain is: **`run_patchcore_raw.sh` succeeds → unified CSVs exist → `reproduce_app_patchcore_tta.sh` completes analyze and copies figures.**
