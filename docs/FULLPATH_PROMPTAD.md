# PromptAD Stage 1 — model evidence (`run_promptad_raw.sh`)

Scope: **PromptAD only** (no PaDiM/PatchCore changes here). **`scripts/reproduce_*` do not call this script** — two-stage layout: Stage 1 extracts raw evidence; Stage 2 section scripts consume cached/stub inputs separately.

---

## Modes (`PROMPTAD_MODE`)

Comma-separated tokens (default **`export`**):

| Token | Behavior |
|-------|----------|
| **`export`** | Scan `PROMPTAD_OUTPUT_ROOT` for existing `CLS-*-per_sample.csv` files and write **`outputs/cached_results/raw_scores/promptad/unified_raw_scores_{wide,long}.csv`** (+ `manifest.json`). **No training or inference** — preferred for stable CI / replays when artifacts already exist. |
| **`train`** | Loop `train_cls.py` for each `(PROMPTAD_DATASETS × PROMPTAD_CLASSES × PROMPTAD_SHOTS × PROMPTAD_SEEDS)` with `--root-dir "$PROMPTAD_OUTPUT_ROOT"`. **Heavy**; requires GPU and data where upstream loaders expect it (see below). |
| **`infer`** | Loop `test_cls.py` with the same Cartesian product. |

Examples: `export` · `infer,export` · `train,infer,export`.

---

## Environment variables

| Variable | When required | Meaning |
|----------|---------------|---------|
| `FULL_RUN` | always | Must be `1` or the script exits (prevents accidental long runs). |
| `PROMPTAD_OUTPUT_ROOT` | always | PromptAd `--root-dir` for train/infer, **or** directory tree to scan for **export** (e.g. `.../result_round1`). |
| `PROMPTAD_DATA_ROOT` | `train` or `infer` in mode | Must exist as a directory; used only to **validate** intent — upstream `datasets/mvtec.py` / `visa.py` still use **fixed** paths (`~/datasets/mvtec`, `~/datasets/pro_visa`) unless you symlink. |
| `PROMPTAD_RAW_OUT` | optional | Output dir for unified CSVs (default: `outputs/cached_results/raw_scores/promptad`). |
| `PROMPTAD_DATASETS` | train/infer required; optional for export filter | Comma list, e.g. `mvtec`. |
| `PROMPTAD_CLASSES` | train/infer required; optional export filter | Comma list, e.g. `bottle,carpet`. |
| `PROMPTAD_SHOTS` | train/infer required; optional export filter | Comma ints, e.g. `1,2,4`. |
| `PROMPTAD_SEEDS` | train/infer required; optional export filter | Comma ints, e.g. `111,222`. |
| `PROMPTAD_GPU` | optional | Sets `CUDA_VISIBLE_DEVICES` (default `0`); train/test use `--gpu-id 0` inside the visible device. |
| `PROMPTAD_TRAIN_EXTRA_ARGS` | optional | Space-separated tokens appended **only** to `train_cls.py` (e.g. `--Epoch 3 --batch-size 8 --eval-freq 1`). |
| `PROMPTAD_INFER_EXTRA_ARGS` | optional | Space-separated tokens appended **only** to `test_cls.py`. |
| `PROMPTAD_EXTRA_ARGS` | optional | **Deprecated alias:** if set and `PROMPTAD_TRAIN_EXTRA_ARGS` is unset, appended **only** to `train_cls.py` (infer does not accept `--Epoch` / `--eval-freq`). |
| `PROMPTAD_RESUME` | optional | If `1`, skip **train** and **infer** for a cell when the expected `CLS-*-per_sample.csv` already exists under `PROMPTAD_OUTPUT_ROOT` (safe resume after partial runs). Default `0` re-runs train/infer. |
| `PROMPTAD_FAIL_FAST` | optional | If `1`, stop the grid on the first train or infer failure. Default `0`: record the failure, continue other cells, then still run **export** so successful settings contribute to unified raw. |
| `PROMPTAD_STATUS_DIR` | optional | Directory for run status (default: `$REPO_ROOT/outputs/promptad_fullpath_status`). |
| `PROMPTAD_RUN_STATUS_CSV` | optional | Per-cell status log (default: `$PROMPTAD_STATUS_DIR/promptad_run_status.csv`). |
| `PROMPTAD_EXIT_OK_ON_PARTIAL` | optional | If `1`, exit **0** after the loop even when some train/infer cells failed (export may still succeed on a subset). Default: exit **1** if any cell failed so CI surfaces partial grids. |

---

## Data path caveat (partial / blocker)

Upstream PromptAD resolves MVTec/VisA image roots inside `external/PromptAD/datasets/*.py` (e.g. `~/datasets/mvtec`). **`PROMPTAD_DATA_ROOT` does not rewrite those modules** without symlinks or a future small upstream hook. **Export-only** mode avoids this entirely.

---

## Section 3.1.1 (Stage 2)

After unified raw files exist, run:

```bash
bash src/experiments/sec3_promptad_observation/run.sh
```

This executes **`analyze_sec3_promptad_from_raw.py`** only (tables + figures). It does **not** call `run_promptad_raw.sh`. If files are missing, set **`SEC3_PROMPTAD_ALLOW_STUB=1`** only for non-empirical bundled CSV.

---

## Unified raw schema (reserved for downstream)

Wide and long tables include at least: `sample_id`, `label`, `fused_score`, `semantic_score`, `visual_score`, `condition`, `condition_score` (long: per-view `semantic` / `visual` / `fused`; wide: `image_level` + harmonic as `condition_score`), `dataset`, `category`, `shot`, `seed`, `config`, `image_path`. Section **3.1.1 / 4 / Appendix C/G** are **not** refactored in this pass; they may later join on `sample_id`, `(dataset, category, shot, seed)`, and paths.

---

## Commands

**A) Export only (recommended when `*-per_sample.csv` already exist)**

```bash
export PROMPTAD_OUTPUT_ROOT=/path/to/promptad_result_round1_or_mixed_tree
# optional filters:
# export PROMPTAD_DATASETS=mvtec
# export PROMPTAD_CLASSES=bottle
# export PROMPTAD_SHOTS=1
# export PROMPTAD_SEEDS=111

FULL_RUN=1 PROMPTAD_MODE=export bash scripts/run_promptad_raw.sh
```

**B) Train + infer + export (expert; data + GPU required)**

```bash
export PROMPTAD_OUTPUT_ROOT=/path/to/writable_promptad_root
export PROMPTAD_DATA_ROOT=/path/to/mvtec_parent   # for validation; symlink to ~/datasets/mvtec as needed
export PROMPTAD_DATASETS=mvtec
export PROMPTAD_CLASSES=bottle
export PROMPTAD_SHOTS=1
export PROMPTAD_SEEDS=111

FULL_RUN=1 PROMPTAD_MODE=train,infer,export bash scripts/run_promptad_raw.sh
```

**C) Minimal full-path smoke (verified 2026-05-12 on this host — not the 81-setting paper grid)**

| Item | Value |
|------|--------|
| **Label** | **minimal full-path verified** — *not* full 81-setting reproduction |
| Dataset / class / shot / seeds | `mvtec` / `bottle` / `1` / `111,222` |
| Data on disk | `/home/zju/datasets/mvtec` (same layout as `~/datasets/mvtec` here — **no symlink required** on this machine) |
| Output root | `PROMPTAD_OUTPUT_ROOT=$REPO_ROOT/outputs/promptad_fullpath_minimal` |
| Train budget | `PROMPTAD_TRAIN_EXTRA_ARGS="--Epoch 3 --batch-size 8 --eval-freq 1 --num-workers 2"` (short run; default upstream `Epoch` is 100) |
| GPU | `PROMPTAD_GPU=0` (RTX 3090 in verification run) |

```bash
cd "$REPO_ROOT"
mkdir -p outputs/promptad_fullpath_minimal
FULL_RUN=1 \
  PROMPTAD_MODE=train,infer,export \
  PROMPTAD_OUTPUT_ROOT="$REPO_ROOT/outputs/promptad_fullpath_minimal" \
  PROMPTAD_DATA_ROOT=/home/zju/datasets/mvtec \
  PROMPTAD_DATASETS=mvtec \
  PROMPTAD_CLASSES=bottle \
  PROMPTAD_SHOTS=1 \
  PROMPTAD_SEEDS=111,222 \
  PROMPTAD_GPU=0 \
  PROMPTAD_TRAIN_EXTRA_ARGS="--Epoch 3 --batch-size 8 --eval-freq 1 --num-workers 2" \
  bash scripts/run_promptad_raw.sh
```

Produces real `CLS-mvtec-bottle-k1-seed{111,222}-per_sample.csv` under `outputs/promptad_fullpath_minimal/mvtec/k_1/csv/` and unified raw under `outputs/cached_results/raw_scores/promptad/`. Stage 2: set `SEC3_PROMPTAD_MAX_PAIRS_PER_SETTING` / `SEC4_MAX_PAIRS_PER_SETTING` if needed, then `bash scripts/reproduce_sec3_promptad.sh` and `bash scripts/reproduce_sec4_systematic.sh`.

The directory `outputs/promptad_fullpath_minimal/` is listed in `.gitignore` — **do not commit** weights or run trees.

**D) Small-scale extension — five seeds (verified 2026-05-12 — not 81-setting)**

| Item | Value |
|------|--------|
| **Label** | **small-scale PromptAD full-path verified** — *not* full 81-setting reproduction |
| Seeds | `111,222,333,444,555` (same class/shot/data/GPU/train args as **(C)**) |
| Output root | `PROMPTAD_OUTPUT_ROOT=$REPO_ROOT/outputs/promptad_fullpath_bottle_k1_5seeds` |

```bash
cd "$REPO_ROOT"
mkdir -p outputs/promptad_fullpath_bottle_k1_5seeds
FULL_RUN=1 \
  PROMPTAD_MODE=train,infer,export \
  PROMPTAD_OUTPUT_ROOT="$REPO_ROOT/outputs/promptad_fullpath_bottle_k1_5seeds" \
  PROMPTAD_DATA_ROOT=/home/zju/datasets/mvtec \
  PROMPTAD_DATASETS=mvtec \
  PROMPTAD_CLASSES=bottle \
  PROMPTAD_SHOTS=1 \
  PROMPTAD_SEEDS=111,222,333,444,555 \
  PROMPTAD_GPU=0 \
  PROMPTAD_TRAIN_EXTRA_ARGS="--Epoch 3 --batch-size 8 --eval-freq 1 --num-workers 2" \
  bash scripts/run_promptad_raw.sh
```

Then refresh Sec 3 / Sec 4 from unified raw (overwrites `outputs/cached_results/raw_scores/promptad/`):

```bash
export SEC3_PROMPTAD_MAX_PAIRS_PER_SETTING=200000
export SEC4_MAX_PAIRS_PER_SETTING=200000
bash scripts/reproduce_sec3_promptad.sh
bash scripts/reproduce_sec4_systematic.sh
```

**Recorded (2026-05-12):** **5** `CLS-*-per_sample.csv`; unified wide **415** rows; Sec 3 `pairwise_metrics` **6300** rows; Sec 4 **`near_auroc_candidate_pairs.csv`** **10** rows; `reproduce_main.sh` **exit 0**. Directory **`outputs/promptad_fullpath_bottle_k1_5seeds/`** is **gitignored**.

---

## (E) Large-scale grid — resume, status, and fail policy

**Not** the full **81-setting** paper grid by default; this section documents how to run **many** `(dataset × class × shot × seed)` cells safely.

### Status file

For `PROMPTAD_MODE` that includes **train** and/or **infer**, the script initializes (truncates) a CSV at:

`outputs/promptad_fullpath_status/promptad_run_status.csv` (override with `PROMPTAD_RUN_STATUS_CSV`).

**Columns:** `dataset`, `category`, `shot`, `seed`, `train_status`, `infer_status`, `export_status`, `per_sample_path`, `error_message`, `start_time`, `end_time`.

**Typical status values:** `ok`, `failed`, `n/a` (mode not used), `skipped_existing` (resume skip), `skipped_train_failed`, `skipped_fail_fast`. After a successful batch **export**, `finalize-export` rewrites `export_status` from `pending` to `ok` or `missing_per_sample` using on-disk paths.

### Resume

```bash
export PROMPTAD_RESUME=1
# same PROMPTAD_OUTPUT_ROOT, datasets/classes/shots/seeds as the partial run
FULL_RUN=1 PROMPTAD_MODE=train,infer,export ... bash scripts/run_promptad_raw.sh
```

Cells that already have the expected `CLS-*-per_sample.csv` are skipped for train and infer; the final **export** still scans the tree and emits unified raw for all matching files.

### Failure policy

| Variable | Effect |
|----------|--------|
| Default | Continue after train/infer failures; append status rows; run **export** once at the end. |
| `PROMPTAD_FAIL_FAST=1` | Stop the grid on the first failing train or infer step (partial status CSV). |
| `PROMPTAD_EXIT_OK_ON_PARTIAL=1` | Exit code **0** even if some cells failed (use for ad-hoc inspection; strict CI should omit this). |

### Helper

`scripts/promptad_run_status.py` implements `init`, `append`, and `finalize-export` for the status CSV (invoked by `run_promptad_raw.sh`; not PromptAD training code).

**Status snapshot:** each `bash scripts/run_promptad_raw.sh` invocation that runs train/infer **truncates** `promptad_run_status.csv` at the start of the grid. Copy it aside if you need a permanent record before re-launching.

**Pipelines:** do not pipe Stage 1 stdout into `head` while training is in progress — the writer can receive **SIGPIPE** and kill the run early.

---

## (F) Medium-scale full-path — 27 settings (verified on host)

**Status label:** **`medium-scale PromptAD full-path verified`** — **not** the full **81-setting** reproduction.

| Item | Value |
|------|--------|
| Grid | `mvtec` × classes **`bottle,cable,capsule`** × shots **`1,2,4`** × seeds **`111,222,333`** → **27** cells |
| Output root | `PROMPTAD_OUTPUT_ROOT=$REPO_ROOT/outputs/promptad_fullpath_mvtec_3cls_3shots_3seeds` |
| Train | `PROMPTAD_TRAIN_EXTRA_ARGS="--Epoch 3 --batch-size 8 --eval-freq 1 --num-workers 2"` |
| Data | `/home/zju/datasets/mvtec` (visible as `~/datasets/mvtec` on the verification host) |

```bash
cd "$REPO_ROOT"
mkdir -p outputs/promptad_fullpath_mvtec_3cls_3shots_3seeds
FULL_RUN=1 \
  PROMPTAD_MODE=train,infer,export \
  PROMPTAD_OUTPUT_ROOT="$REPO_ROOT/outputs/promptad_fullpath_mvtec_3cls_3shots_3seeds" \
  PROMPTAD_DATA_ROOT=/home/zju/datasets/mvtec \
  PROMPTAD_DATASETS=mvtec \
  PROMPTAD_CLASSES=bottle,cable,capsule \
  PROMPTAD_SHOTS=1,2,4 \
  PROMPTAD_SEEDS=111,222,333 \
  PROMPTAD_GPU=0 \
  PROMPTAD_TRAIN_EXTRA_ARGS="--Epoch 3 --batch-size 8 --eval-freq 1 --num-workers 2" \
  bash scripts/run_promptad_raw.sh
# After an interruption, re-run the same env with PROMPTAD_RESUME=1 to skip finished cells (see section (E)).
```

Then (overwrites default unified raw under `outputs/cached_results/raw_scores/promptad/`):

```bash
export SEC3_PROMPTAD_MAX_PAIRS_PER_SETTING=200000
export SEC4_MAX_PAIRS_PER_SETTING=200000
bash scripts/reproduce_sec3_promptad.sh
bash scripts/reproduce_sec4_systematic.sh
bash scripts/reproduce_main.sh
```

**Recorded metrics (2026-05-12, this host):**

| Metric | Value |
|--------|--------|
| `CLS-*-per_sample.csv` (excluding instability sidecars) | **27** (= 3×3×3) |
| `promptad_run_status.csv` data rows | **27** (all `train_status`/`infer_status`/`export_status` **ok** after finalize; first cell used **`PROMPTAD_RESUME=1`** skip where artifact already existed) |
| `unified_raw_scores_wide.csv` logical rows | **3285** (3286 lines incl. header) |
| `unified_raw_scores_long.csv` logical rows | **9855** (9856 lines incl. header) |
| Sec 3 `pairwise_metrics.csv` data rows | **81927** |
| Sec 4 `near_auroc_candidate_pairs.csv` candidate pairs | **21** (22 lines incl. header) |
| `bash scripts/reproduce_sec3_promptad.sh` / `reproduce_sec4_systematic.sh` / `reproduce_main.sh` | **exit 0** (with `SEC3_PROMPTAD_MAX_PAIRS_PER_SETTING=200000`, `SEC4_MAX_PAIRS_PER_SETTING=200000`) |

Details: **`docs/END_TO_END_VALIDATION.md`** §13. Artifacts under **`outputs/promptad_fullpath_mvtec_3cls_3shots_3seeds/`** and **`outputs/promptad_fullpath_status/`** are **gitignored**.

---

## Verification status (this repo)

**Train/infer/export on this host:** **(C)** two-seed minimal loop, **(D)** five-seed extension, and **(F)** **27-setting** medium grid (§13 in **`docs/END_TO_END_VALIDATION.md`**) completed with short `Epoch=3`, real `CLS-*-per_sample.csv`, unified raw + pairwise Sec 3 / Sec 4 + `reproduce_main.sh` (with `SEC3_PROMPTAD_MAX_PAIRS_PER_SETTING=200000`, `SEC4_MAX_PAIRS_PER_SETTING=200000`). **Not** a reproduction of the full multi-class multi-seed paper grid (**81 settings**). Large grids: **`docs/FULLPATH_PROMPTAD.md`** section **(E)** (`PROMPTAD_RESUME`, status CSV, `PROMPTAD_FAIL_FAST`).

**Default clone / CI:** still has **no** bundled full `result_round1` tree; **export-only** remains the light path when you already have `CLS-*-per_sample.csv` elsewhere. **Interface checks:** `bash -n scripts/run_promptad_raw.sh`; `FULL_RUN=1` without `PROMPTAD_OUTPUT_ROOT` fails fast (`:?`); **`PROMPTAD_DATA_ROOT` is required when `PROMPTAD_MODE` includes `train` or `infer`** (export-only does not need it). Upstream loaders use **`~/datasets/mvtec`** unless symlinked.
