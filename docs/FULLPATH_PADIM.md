# PaDiM full-path reproduction (Protocol B raw → Section 3.1.2 & Appendix E)

Scope: **PaDiM only** (no PromptAD). The default **fast path** still copies bundled stubs from `samples/fastpath/`.

### (A) Debug / smoke / demo — `PADIM_PROFILE=debug`

**Purpose:** quick wiring checks and **small** Protocol B sweeps (default **five** MVTec classes × your `PADIM_SEEDS` × `PADIM_BACKBONES`). **Not** a paper-level benchmark label. When `PADIM_CLASSES` is **unset**, `scripts/run_padim_raw.sh` fills a **smoke five-class** list (`bottle,cable,capsule,screw,toothbrush`). The historical **`outputs/padim_protocol_b_smallgrid/`** run on this repo is documented below as an example of this tier.

### (B) Canonical paper-level FULL_RUN — `PADIM_PROFILE=paper` (default)

**Definition (MVTec Protocol B):** `PADIM_DATASET=mvtec` (default), **`PADIM_BACKBONES=resnet18`**, **`PADIM_SEEDS=111,222,333,444,555`**, and **all fifteen** MVTec AD object classes:

`bottle`, `cable`, `capsule`, `carpet`, `grid`, `hazelnut`, `leather`, `metal_nut`, `pill`, `screw`, `tile`, `toothbrush`, `transistor`, `wood`, `zipper`.

When **`PADIM_CLASSES` is unset** and `PADIM_PROFILE=paper` (the default), `run_padim_raw.sh` auto-fills that **MVTec-15** list. **Explicit `PADIM_CLASSES` always wins** over `PADIM_PROFILE`. For **`visa`** or any non-`mvtec` dataset, you must set `PADIM_CLASSES` yourself (`paper` cannot infer the class list).

**Two-stage alignment:** `scripts/run_padim_raw.sh` is **Stage 1** (Protocol B jobs + aggregation). `scripts/reproduce_sec3_padim.sh` and `scripts/reproduce_app_padim_representation.sh` are **Stage 2** only — they **never** invoke Stage 1; **`FULL_RUN=1` on `reproduce_*` does not run** `run_padim_raw.sh`. **`FULL_RUN=1`** is still required **on Stage 1** (`run_padim_raw.sh`) for real inference.

### Appendix E — scope (important)

**`FULL_RUN=1` in `app_padim_representation/run.sh` does *not* claim full Appendix E / seed-killer reproduction.** It only reuses the same Protocol B job tree as Section 3.1.2 and writes **minimal raw-derived summaries**:

- `mechanism_from_raw.csv` — a **two-row** global summary derived from **your** `PADIM_SEEDS` list (mean / max pairwise |ΔAUROC| across settings).  
- This is **not** equivalent to the full multi-phase **`padim_seed_killer_evidence_pipeline.py`** (phase1 metrics mining, phase2 bash grid, phase3 risk–coverage killer selection, appendix pair figures). Those remain **out of band** unless you run upstream scripts manually with their expected directory layout.

Section 3.1.2 full path consumes **`marginal_protocol_b.csv`** + optional scatter from the same raw aggregation.

---

Design goals:

- **No accidental silent class lists:** `PADIM_PROFILE` (`debug` \| `paper`, default **`paper`**) supplies **`PADIM_CLASSES` only when it is unset**. Explicit `PADIM_CLASSES` is never overwritten. `FULL_RUN=1` still runs the Cartesian product **`PADIM_CLASSES` × `PADIM_BACKBONES` × `PADIM_SEEDS`** only (wrapper `scripts/run_padim_raw.sh`). The legacy **`run_padim_seed_killer_one_click.sh`** sweep is **not** wired into `run.sh` anymore.
- **Unified raw scores** under `outputs/cached_results/raw_scores/padim/`.
- **Downstream CSVs** regenerated from those jobs: `marginal_protocol_b.csv` (sec3) and `mechanism_from_raw.csv` (appendix E summary).

---

## Environment

- **`PADIM_PROFILE`:** `debug` \| `paper` (default **`paper`**). See **§ Profiles** at the top. Used only when **`PADIM_CLASSES` is empty** (then `run_padim_raw.sh` sets classes; explicit `PADIM_CLASSES` always wins).
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
# Canonical paper run (auto MVTec-15 when PADIM_CLASSES unset):
export PADIM_PROFILE=paper         # optional; default is paper
export PADIM_BACKBONES=resnet18
export PADIM_SEEDS=111,222,333,444,555
# export PADIM_CLASSES=bottle,cable,...   # optional; when set, overrides profile auto-fill
# export PADIM_PROFILE=debug              # smoke: auto five classes when PADIM_CLASSES unset
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

## Debug / demo validation — example run (**not** canonical)

**Label:** **debug validation only** — five MVTec classes × five seeds × `resnet18` (25 Protocol B jobs). Used for smoke tests and figure-density checks; **do not** treat this as the paper-level Protocol B benchmark.

**Command (repo root):** set `PADIM_PROFILE=debug` and omit `PADIM_CLASSES` (auto smoke-five), or set `PADIM_CLASSES` explicitly (still respected).

```bash
export FULL_RUN=1
export PADIM_PROFILE=debug
export PADIM_DATA_ROOT=/path/to/mvtec
export PADIM_OUTPUT_ROOT=/path/to/repo/outputs/padim_protocol_b_smallgrid
export PADIM_DATASET=mvtec
export PADIM_BACKBONES=resnet18
export PADIM_SEEDS=111,222,333,444,555
# export PADIM_CLASSES=bottle,cable,capsule,screw,toothbrush   # optional override

bash scripts/run_padim_raw.sh
bash scripts/reproduce_sec3_padim.sh
python3 src/experiments/sec3_padim_observation/build_paper_style_fig2_padim.py
```

### On-host record (`outputs/padim_protocol_b_smallgrid/`, 2026-05-12)

| Check | Result (this run) |
|--------|-------------------|
| `run_padim_raw.sh` exit | **0** |
| Per-job root | `outputs/padim_protocol_b_smallgrid/protocol_b_jobs/mvtec__<class>__resnet18/<seed>/` |
| `unified_raw_scores_{long,wide}.csv` | under `outputs/cached_results/raw_scores/padim/` |
| `marginal_protocol_b.csv` rows | **25** data rows (+ header) |
| `reproduce_sec3_padim.sh` | **0** (uses cached marginal) |
| `build_paper_style_fig2_padim.py` | **0** → `outputs/figures/sec3_padim/paper_style_fig2.{png,pdf}` |

### Coverage audit (debug example; `marginal_protocol_b.csv`)

| Field | Value |
|--------|--------|
| **Datasets** | **1** (`mvtec`) |
| **Categories** | **5** (`bottle`, `cable`, `capsule`, `screw`, `toothbrush`) |
| **Backbones** | **1** (`resnet18`) |
| **Seeds** | **5** (`111`, `222`, `333`, `444`, `555`) |
| **Total runs** (rows in marginal CSV) | **25** |
| **Category × seed completeness** | **Full 5×5 grid** — each `(category, seed)` appears **exactly once** |

**Numeric gates** (≥5 categories, ≥5 seeds, ≥25 runs): **met** for this debug grid — still **not** canonical **15-class** coverage. A `paper_style_fig2_summary.json` from `build_paper_style_fig2_padim.py` on this grid records **`small_grid": true`** (75-run canonical tag is **not** applicable).

`outputs/padim_protocol_b_smallgrid/` is listed in `.gitignore` — **do not commit** job trees or large `outputs/` artifacts.

---

## Canonical paper-level FULL_RUN (MVTec **15** classes × **5** seeds × `resnet18`)

**Definition:** `PADIM_DATASET=mvtec` (default), `PADIM_BACKBONES=resnet18`, `PADIM_SEEDS=111,222,333,444,555`, **`PADIM_PROFILE=paper`** (default), **`PADIM_CLASSES` unset** → `run_padim_raw.sh` injects the full **MVTec-15** class list above → **15 × 1 × 5 = 75** Protocol B jobs, then aggregation to `outputs/cached_results/raw_scores/padim/` and `marginal_protocol_b.csv`.

**Command sketch:**

```bash
export FULL_RUN=1
export PADIM_PROFILE=paper    # default; omit if acceptable
export PADIM_DATA_ROOT=/path/to/mvtec
export PADIM_OUTPUT_ROOT=/path/to/writable_root   # e.g. outputs/padim_protocol_b_mvtec15x5/
export PADIM_BACKBONES=resnet18
export PADIM_SEEDS=111,222,333,444,555
# Do NOT set PADIM_CLASSES for canonical auto-fill on mvtec.

bash scripts/run_padim_raw.sh
bash scripts/reproduce_sec3_padim.sh
python3 src/experiments/sec3_padim_observation/build_paper_style_fig2_padim.py
```

**Rough wall-time (single GPU, sequential jobs):** scale from the debug example (**~6 min** for **25** jobs) → **~18–25 min** for **75** jobs, plus aggregation — budget **~25–40 minutes** if image counts vary by class.

**On-host check (2026-05-13):** `PADIM_OUTPUT_ROOT=outputs/padim_protocol_b_mvtec15x5/` under the repo, **~19 min** wall clock (RTX 3090), `marginal_protocol_b.csv` **75** data rows, unified raw **8625** wide rows, Sec 3.2 figures refreshed under `outputs/figures/sec3_padim/` (`padim_marginal_scatter.png`, `paper_style_fig2.{png,pdf}` from `build_paper_style_fig2_padim.py`).

---

## Runtime

Depends on image count, backbone (`wide_resnet50_2` ≫ `resnet18`), and seeds. Expect **minutes per (class, arch, seed)** on a single GPU for a single MVTec class; full benchmarks are intentionally **not** triggered unless you expand `PADIM_*` lists.

---

## Verification / blockers (this repo)

A **debug-profile** Protocol B pass (five MVTec classes × five seeds × `resnet18`) was executed on-host; see **§ Debug / demo validation** for artifacts. **Canonical paper-level** coverage is **15 classes × 5 seeds** (`PADIM_PROFILE=paper` with unset `PADIM_CLASSES` on `mvtec`) and is **documented** here — run locally when data and GPU are available.

Generic blockers: invalid `PADIM_DATA_ROOT` → upstream dataset / forward failures. **`PADIM_BACKBONES`** and **`PADIM_SEEDS`** remain required. **`PADIM_CLASSES`** is optional only when profile auto-fill applies (`mvtec` + `paper` or `debug`); for **non-mvtec** datasets, set **`PADIM_CLASSES`** explicitly.
