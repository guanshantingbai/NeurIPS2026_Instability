# Appendix E PaDiM Representation-Level Conditions

Reproduces PaDiM appendix analyses:

- near-AUROC selection behavior;
- mechanism chain;
- failure signal comparison.

**Fast path:** copies bundled `mechanism_stub.csv` from `samples/fastpath/`.

**Raw-derived (Stage 1 then Stage 2):** run `FULL_RUN=1 bash scripts/run_padim_raw.sh`, then `bash scripts/reproduce_app_padim_representation.sh` to consume `mechanism_from_raw.csv` (minimal cross-seed summary — **not** full seed-killer). See **`docs/FULLPATH_PADIM.md`** and **`docs/REPRODUCE.md`**.
