# Initial Release Plan

This document tracks the transition from a submission-time public repository to a reproducible research release.

## Phase 0 (current)

- Repository is public and accessible.
- Core code directories are present.
- Large experiment artifacts are excluded via `.gitignore`.

## Phase 1

- Add unified environment specification (`requirements.txt` or `environment.yml`).
- Add dataset preparation instructions.
- Add minimal runnable smoke test scripts.

## Phase 2

- Map each paper table/figure to a script entry point.
- Add expected metric ranges for key experiments.
- Add CI checks for lint and smoke tests.

## Notes

- The current release prioritizes accessibility and code availability near deadline.
- Reproducibility hardening is planned as incremental updates.
