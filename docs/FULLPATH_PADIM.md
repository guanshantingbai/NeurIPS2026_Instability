# PaDiM full-path reproduction (Protocol B raw → Section 3.1.2 & Appendix E)

Scope: **PaDiM only** (no PromptAD). The default **fast path** still copies bundled stubs from `samples/fastpath/`.

**Two-stage alignment:** `scripts/run_padim_raw.sh` is **Stage 1** (Protocol B jobs + aggregation). `scripts/reproduce_sec3_padim.sh` and `scripts/reproduce_app_padim_representation.sh` are **Stage 2** only — they **never** invoke Stage 1; **`FULL_RUN=1` on `reproduce_*` does not run** `run_padim_raw.sh`. **`FULL_RUN=1`** is still required **on Stage 1** (`run_padim_raw.sh`) for real inference.

### Appendix E — scope (important)

**`FULL_RUN=1` in `app_padim_representation/run.sh` does *not* claim full Appendix E / seed-killer reproduction.** It only reuses the same Protocol B job tree as Section 3.1.2 and writes **minimal raw-derived summaries**:

- `mechanism_from_raw.csv` — a **two-row** global summary derived from **your** `PADIM_SEEDS` list (mean / max pairwise |ΔAUROC| across settings).  
- This is **not** equivalent to the full multi-phase **`padim_seed_killer_evidence_pipeline.py`** (phase1 metrics mining, phase2 bash grid, phase3 risk–coverage killer selection, appendix pair figures). Those remain **out of band** unless you run upstream scripts manually with their expected directory layout.

Section 3.1.2 full path consumes **`marginal_protocol_b.csv`** + optional scatter from the same raw aggregation.

---

Design goals:

- **No accidental full-benchmark runs:** `FULL_RUN=1` calls `external/PaDiM-Anomaly-Detection-Localization-master/padim_protocol_b_one_run.py` only for the Cartesian product of **`PADIM_CLASSES` × `PADIM_BACKBONES` × `PADIM_SEEDS`** (wrapper `scripts/run_padim_raw.sh`). The legacy **`run_padim_seed_killer_one_click.sh`** sweep is **not** wired into `run.sh` anymore.
- **Unified raw scores** under `outputs/cached_results/raw_scores/padim/`.
- **Downstream CSVs** regenerated from those jobs: `marginal_protocol_b.csv` (sec3) and `mechanism_from_raw.csv` (appendix E summary).

---

## Environment

- Python 3, PyTorch, torchvision, scikit-learn, pandas, numpy, matplotlib, Pillow (same as upstream PaDiM).
- **GPU** recommended (`padim_protocol_b_one_run` selects CUDA when available). `PADIM_GPU` maps to **`CUDA_VISIBLE_DEVICES`** so upstream `cuda:0` binds to the chosen physical GPU.

---

## Data

- **`PADIM_DATA_ROOT`**: MVTec parent (class subfolders) or VisA root as expected by `datasets/mvtec.py` / `datasets/visa.py` in the external tree.
- **`PADIM_OUTPUT_ROOT`**: any writable directory for per-seed artifacts (`protocol_b_jobs/...`). Prefer **scratch outside the repo**, or `outputs/padim_protocol_b_jobs` (gitignored) if you keep everything local.

VisA × `wide_resnet50_2` often needs extra flags (see upstream scripts), e.g. in **`PADIM_EXTRA_ARGS`**:

```bash
export PADIM_EXTRA_ARGS="--cov-float32 --max-train-images 350"
```

---

## Commands

### Raw Protocol B + unified export + cached summaries

```bash
cd /path/to/NeurIPS2026
export PADIM_DATA_ROOT=/path/to/mvtec_or_visa
export PADIM_OUTPUT_ROOT=/path/to/writable_root   # e.g. /tmp/padim_pb or outputs/padim_protocol_b_jobs
export PADIM_DATASET=mvtec          # optional; default mvtec
export PADIM_CLASSES=bottle
export PADIM_BACKBONES=resnet18
export PADIM_SEEDS=444,555
# export PADIM_GPU=0
# export PADIM_FORCE=1            # overwrite existing per_sample.csv
# export PADIM_EXTRA_ARGS="--cov-float32"

FULL_RUN=1 bash scripts/run_padim_raw.sh
```

### Section 3.1.2 — Stage 2 only

```bash
bash scripts/reproduce_sec3_padim.sh
```

Runs `src/experiments/sec3_padim_observation/run.sh`: uses `marginal_protocol_b.csv` if present, else bundled stubs — **no** `run_padim_raw.sh`.

### Appendix E — Stage 2 only

```bash
bash scripts/reproduce_app_padim_representation.sh
```

Consumes **`mechanism_from_raw.csv`** when present (written by Stage 1 aggregation), else bundled stub — **no** `run_padim_raw.sh`. The CSV remains a **raw-derived partial** summary, **not** the full seed-killer pipeline.

---

## Outputs

| Path | Content |
|------|---------|
| `PADIM_OUTPUT_ROOT/protocol_b_jobs/<dataset>__<class>__<arch>/<seed>/per_sample.csv` | Upstream per-image Protocol B scores |
| `.../summary.json` | AUROC, instability, counts |
| `outputs/cached_results/raw_scores/padim/unified_raw_scores_long.csv` | Long-form unified raw (view_id = `s0`/`s1`/`s2`/`s_fused`) |
| `outputs/cached_results/raw_scores/padim/unified_raw_scores_wide.csv` | One row per image + `view_scores` JSON |
| `outputs/cached_results/sec3_padim/marginal_protocol_b.csv` | `setting,seed,auroc,instability` for sec3 |
| `outputs/cached_results/app_padim_representation/mechanism_from_raw.csv` | Global cross-seed ΔAUROC summary (2 rows) |
| `outputs/figures/sec3_padim/padim_marginal_scatter.png` | Optional quick scatter when marginal CSV exists |

All under `outputs/` remains **gitignored** except `.gitkeep` skeletons — do not commit large runs.

---

## Runtime

Depends on image count, backbone (`wide_resnet50_2` ≫ `resnet18`), and seeds. Expect **minutes per (class, arch, seed)** on a single GPU for a single MVTec class; full benchmarks are intentionally **not** triggered unless you expand `PADIM_*` lists.

---

## Verification / blockers (this repo)

End-to-end Protocol B was **not** executed here (no bundled MVTec/VisA install). With an invalid `PADIM_DATA_ROOT`, upstream fails during dataset load or model forward — **partial** until you provide real data and GPU.

If `PADIM_CLASSES` / `PADIM_BACKBONES` / `PADIM_SEEDS` are unset, `run_padim_raw.sh` exits with a clear `:?` error (prevents silent “full grid” runs).
