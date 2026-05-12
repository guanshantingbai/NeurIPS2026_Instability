# Reproduce Guide

This repository reproduces results by paper sections using a **two-stage** layout.

## Stage 1 — model-level evidence extraction

Runs **real** upstream inference / scoring (GPU + data as applicable). **Orchestration scripts `scripts/reproduce_*.sh` never invoke Stage 1.**

| Model | Script | Typical outputs |
|-------|--------|-------------------|
| PatchCore | `FULL_RUN=1 bash scripts/run_patchcore_raw.sh` | `outputs/cached_results/raw_scores/patchcore/` (`patchcore_tta_scores.csv`, `unified_raw_scores_long.csv`, …) |
| PaDiM | `FULL_RUN=1 bash scripts/run_padim_raw.sh` | `PADIM_OUTPUT_ROOT/protocol_b_jobs/…`, then aggregated `outputs/cached_results/raw_scores/padim/`, `marginal_protocol_b.csv`, `mechanism_from_raw.csv` |
| PromptAD | `FULL_RUN=1 bash scripts/run_promptad_raw.sh` | `outputs/cached_results/raw_scores/promptad/` (default **export** from existing `*-per_sample.csv`; optional **train**/**infer**) — **`docs/FULLPATH_PROMPTAD.md`** |

See **`docs/MODEL_REPRODUCTION.md`**, **`docs/FULLPATH_PATCHCORE.md`**, and **`docs/FULLPATH_PADIM.md`** for environment variables and caveats.

## Stage 2 — section-level analysis reproduction

`scripts/reproduce_*.sh` and section `run.sh` files **only** read existing **raw scores** or **cached** CSV/figures. They **do not** call `run_patchcore_raw.sh` or `run_padim_raw.sh`, and **`FULL_RUN=1` does not change that.**

| Section | Stage 2 script |
|---------|----------------|
| Appendix F (PatchCore + TTA) | `bash scripts/reproduce_app_patchcore_tta.sh` |
| Section 3.1.2 (PaDiM) | `bash scripts/reproduce_sec3_padim.sh` |
| Appendix E (PaDiM representation) | `bash scripts/reproduce_app_padim_representation.sh` |

### Raw-derived vs fast path (Stage 2)

- **PatchCore Appendix F:** if `outputs/cached_results/raw_scores/patchcore/unified_raw_scores_long.csv` exists, runs **analyze only** (needs `PATCHCORE_DATA_ROOT`, `PATCHCORE_MODELS_RUN` for upstream CLI). Otherwise copies from `result_analysis/patchcore_tta/`. Set **`PATCHCORE_FROM_RAW=1`** to **require** raw evidence and error if it is missing.
- **PaDiM sec3:** if `outputs/cached_results/sec3_padim/marginal_protocol_b.csv` exists, plots only. Else uses bundled stubs under `samples/fastpath/`. Set **`PADIM_FROM_RAW=1`** to require raw-derived artifacts and error if missing (see `docs/FULLPATH_PADIM.md`).
- **PromptAD sec3.1.1:** requires `outputs/cached_results/raw_scores/promptad/unified_raw_scores_wide.csv` or `_long.csv` unless **`SEC3_PROMPTAD_ALLOW_STUB=1`** (bundled stub for demos/CI without raw).
- **PaDiM Appendix E:** if `outputs/cached_results/app_padim_representation/mechanism_from_raw.csv` exists, copies to tables. Else bundled stub. **`PADIM_FROM_RAW=1`** requires that CSV.

## Fast path (default Stage 2)

By default, `scripts/reproduce_main.sh` and `scripts/reproduce_appendix.sh` run **Stage 2** with **bundled stubs** where raw evidence is absent:

- No PatchCore / PaDiM **scoring** from `reproduce_*`.
- Appendix F uses `result_analysis/patchcore_tta/` when unified raw is absent.
- `PYTHONPATH` is set automatically by each `run.sh` / top-level script (repo root).

## Example full flows (two-stage)

**a) PatchCore Appendix F**

```bash
# Stage 1 (GPU + data)
export PATCHCORE_DATA_ROOT=...
export PATCHCORE_MODELS_RUN=...
FULL_RUN=1 bash scripts/run_patchcore_raw.sh

# Stage 2 (analyze + figures from cached raw; needs same env for argparse)
export PATCHCORE_DATA_ROOT=...
export PATCHCORE_MODELS_RUN=...
bash scripts/reproduce_app_patchcore_tta.sh
```

**b) PaDiM Section 3.1.2**

```bash
# Stage 1
export PADIM_DATA_ROOT=...
export PADIM_OUTPUT_ROOT=...
export PADIM_CLASSES=bottle
export PADIM_BACKBONES=resnet18
export PADIM_SEEDS=444,555
FULL_RUN=1 bash scripts/run_padim_raw.sh

# Stage 2
bash scripts/reproduce_sec3_padim.sh
```

**c) PromptAD unified raw (export from existing per-sample CSVs)**

```bash
# Stage 1 (default PROMPTAD_MODE=export)
export PROMPTAD_OUTPUT_ROOT=/path/to/promptad_result_tree   # e.g. result_round1 root
FULL_RUN=1 bash scripts/run_promptad_raw.sh
# -> outputs/cached_results/raw_scores/promptad/
```

## Main Paper

```bash
bash scripts/reproduce_main.sh
```

Section-specific:

```bash
bash scripts/reproduce_sec3_promptad.sh
bash scripts/reproduce_sec3_padim.sh
bash scripts/reproduce_sec4_systematic.sh
```

## Appendix

```bash
bash scripts/reproduce_appendix.sh
```

Section-specific appendix:

```bash
bash src/experiments/app_promptad_generalization/run.sh
bash scripts/reproduce_app_padim_representation.sh
bash scripts/reproduce_app_patchcore_tta.sh
bash src/experiments/app_signal_comparison/run.sh
```

## Input/Output Convention

- Inputs:
  - dataset locations and split metadata are configured in each section `config.yaml`.
  - external model code is referenced through `external/`.
- Outputs:
  - figures: `outputs/figures/<section>/`
  - tables: `outputs/tables/<section>/`
  - cached lightweight results: `outputs/cached_results/<section>/`

## Cleanup

```bash
bash scripts/clean_outputs.sh
```

## Notes

- Some section scripts call external adapters and may require local paths in configs.
- For large experiments, run Stage 1 independently, then Stage 2, and merge artifacts later.
