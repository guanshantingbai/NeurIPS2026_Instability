# Section 3.1.1 PromptAD Observation

This package reproduces PromptAD empirical observations:

- same AUROC, different instability;
- instability distribution;
- instability vs ranking error;
- same AUROC, different risk-coverage.

## Inputs

- PromptAD outputs from external pipeline.
- Local config in `config.yaml`.

## Outputs

- `outputs/figures/sec3_promptad/`
- `outputs/cached_results/sec3_promptad/`

## Run

Default (**fast path**): copies bundled CSV stubs from `samples/fastpath/` and writes `fastpath_done.txt` only.

```bash
bash src/experiments/sec3_promptad_observation/run.sh
```

**Full run:** `FULL_RUN=1 bash src/experiments/sec3_promptad_observation/run.sh` (requires existing PromptAD `result_seed_search` artifacts for `phase3`).
