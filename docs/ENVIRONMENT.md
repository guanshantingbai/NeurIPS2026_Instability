# Environment

## Policy

Use one unified local environment for all section pipelines in this stage.
No multi-env split is required.

## Baseline Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
```

Install dependencies from external projects as needed:

```bash
pip install -r external/PromptAD/requirements.txt || true
pip install -r external/patchcore-inspection/requirements.txt || true
```

PaDiM dependencies can be installed manually based on its scripts if no requirements file exists.

## Runtime Notes

- Prefer running one section at a time for reproducibility bookkeeping.
- Save lightweight CSV and plot outputs in `outputs/`.
- Keep heavy artifacts local and ignored by git.

## Repro Metadata

For each run, store:

- config snapshot;
- command line;
- git commit hash;
- timestamp.
