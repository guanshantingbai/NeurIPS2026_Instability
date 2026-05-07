# NeurIPS2026_Instability

Official repository for the NeurIPS 2026 submission on instability analysis in industrial anomaly detection.

## Current Release Status

This is an initial public release for submission-time code availability.
The repository is being cleaned up and documented incrementally.

Current focus:
- publish core source code;
- exclude large experiment artifacts from version control;
- provide a clear path toward full reproducibility.

## Repository Components

- `PromptAD/`: PromptAD-based pipeline and instability analysis scripts.
- `patchcore-inspection/`: PatchCore baseline code and evaluation scripts.
- `PaDiM-Anomaly-Detection-Localization-master/`: PaDiM baseline code and related experiments.
- `Figures/`: lightweight figure assets used in analysis notes.
- `result_analysis/`: project-level result aggregation scripts/assets.
- `docs/`: repository notes and release planning.

## What Is Excluded in This Initial Version

To keep the repository lightweight and clone-friendly, large generated artifacts are ignored:
- training logs and temporary outputs;
- model checkpoints;
- large result folders;
- packaged figure archives and other large binaries.

See `.gitignore` for exact rules.

## Reproducibility Plan

A progressively improved reproducibility package (environment setup, data preparation, one-click scripts for key tables/figures) will be added in follow-up updates.

## Contact

For questions, please open an issue in this repository.
