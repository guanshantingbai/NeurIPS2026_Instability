# Reproduce Guide

This repository reproduces results by paper sections.

## Fast path (default)

By default, `scripts/reproduce_main.sh` and `scripts/reproduce_appendix.sh` run **only the fast path**:

- No full GPU multi-seed PromptAD or PaDiM pipelines.
- Appendix F copies lightweight CSV/figures from `result_analysis/patchcore_tta/` (must exist).
- `PYTHONPATH` is set automatically by each `run.sh` / top-level script (repo root).

Full model-level reruns require **`FULL_RUN=1`** on the relevant `run.sh` (see section READMEs and `docs/REPRODUCIBILITY_STATUS.md`).

PatchCore model-level steps are documented in **`docs/MODEL_REPRODUCTION.md`** (`scripts/run_patchcore_raw.sh`, env vars, raw score layout).

## Main Paper

Run all main-paper pipelines:

```bash
bash scripts/reproduce_main.sh
```

Run section-specific pipelines:

```bash
bash scripts/reproduce_sec3_promptad.sh
bash scripts/reproduce_sec3_padim.sh
bash scripts/reproduce_sec4_systematic.sh
```

## Appendix

Run all appendix pipelines:

```bash
bash scripts/reproduce_appendix.sh
```

Run section-specific appendix pipelines:

```bash
bash src/experiments/app_promptad_generalization/run.sh
bash src/experiments/app_padim_representation/run.sh
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

Remove generated outputs while keeping tracked examples:

```bash
bash scripts/clean_outputs.sh
```

## Notes

- Some section scripts call external adapters and may require local paths in configs.
- For large experiments, run section scripts independently and merge artifacts later.
