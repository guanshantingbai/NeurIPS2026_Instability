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

**Full run:** `FULL_RUN=1 bash .../run.sh` (long GPU / multi-seed).
