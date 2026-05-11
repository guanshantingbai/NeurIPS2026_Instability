# Dataset Guide

## Principle

Datasets are not versioned in this repository.
Use local paths and keep raw data outside the git workspace when possible.

## Recommended Local Layout

```text
<data_root>/
  mvtec/
  visa/
```

## Configuration Rule

Every experiment package contains a `config.yaml` with:

- `data_root`: absolute or project-relative dataset root.
- `dataset_name`: `mvtec` or `visa`.
- optional section-specific filters (classes, seeds, candidate ids).

## Validation Checklist

Before running scripts:

1. confirm path exists;
2. confirm required classes/subfolders exist;
3. run one section script in dry-run mode first if available.

## Non-Committed Data

Do not commit:

- full datasets;
- checkpoints;
- full training logs;
- raw large-scale intermediate outputs.
