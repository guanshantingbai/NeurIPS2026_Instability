# Section 3.1.2 PaDiM Observation

This package reproduces PaDiM representation-level equivalent conditions:

- marginal subspace readouts;
- instability vs AUROC;
- instability vs error;
- risk-coverage curve.

## Outputs

- `outputs/figures/sec3_padim/`
- `outputs/cached_results/sec3_padim/`

## Run

Default (**fast path**): copies bundled CSV stubs; does **not** run PaDiM Protocol B.

```bash
bash src/experiments/sec3_padim_observation/run.sh
```

**Two-stage:** **Stage 1** — `FULL_RUN=1 bash scripts/run_padim_raw.sh` with `PADIM_*`. **Stage 2** — `bash scripts/reproduce_sec3_padim.sh` (plots from `marginal_protocol_b.csv` or bundled stub; **never** runs Stage 1). See **`docs/FULLPATH_PADIM.md`**, **`docs/REPRODUCE.md`**.
