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

---

## Verification status (this repo)

**Minimal train/infer/export:** verified on **2026-05-12** with configuration **(C)** above: real per-sample CSVs, `unified_raw_scores_{wide,long}.csv`, `manifest.json`, and successful Sec 3 / Sec 4 pairwise pipelines from unified raw (`reproduce_main.sh` exit 0 with `SEC3_PROMPTAD_MAX_PAIRS_PER_SETTING=200000`, `SEC4_MAX_PAIRS_PER_SETTING=200000`). **Not** a reproduction of the full multi-class multi-seed paper grid (81 settings).

**Default clone / CI:** still has **no** bundled full `result_round1` tree; **export-only** remains the light path when you already have `CLS-*-per_sample.csv` elsewhere. **Interface checks:** `bash -n scripts/run_promptad_raw.sh`; `FULL_RUN=1` without `PROMPTAD_OUTPUT_ROOT` fails fast (`:?`); **`PROMPTAD_DATA_ROOT` is required when `PROMPTAD_MODE` includes `train` or `infer`** (export-only does not need it). Upstream loaders use **`~/datasets/mvtec`** unless symlinked.
