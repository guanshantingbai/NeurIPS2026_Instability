# End-to-end validation report

This document records a **single controlled validation pass** from a clean `outputs/` tree. It does **not** claim paper-identical reproduction unless Stage 1 raw evidence was actually present and consumed.

## 1. When and where

| Field | Value |
|--------|--------|
| **Date** | 2026-05-12 |
| **Host** | `linux-pc` (Ubuntu 22.04 kernel 6.8.0-111-generic, x86_64) |
| **Repo root** | `/home/zju/mywork/NeurIPS2026` |
| **Python** | `Python 3.10.17` (`/home/zju/miniconda3/envs/myenv/bin/python3`) |

## 2. Phase 0 — hygiene and static checks

Commands (repo root):

```bash
bash scripts/clean_outputs.sh
bash -n scripts/*.sh
python3 -m compileall -q src
```

| Step | Exit | Notes |
|------|------|--------|
| `clean_outputs.sh` | **0** | Restored `.gitkeep` markers under `outputs/` |
| `bash -n scripts/*.sh` | **0** | All `scripts/*.sh` parse |
| `python3 -m compileall -q src` | **0** | `src/` compiles |

## 3. Phase 1 — raw evidence (Stage 1)

Goal: produce or confirm `outputs/cached_results/raw_scores/{promptad,padim,patchcore}/` unified exports **without** inventing paths on this machine.

### 3.1 PromptAD (`scripts/run_promptad_raw.sh`)

Attempted:

```bash
FULL_RUN=1 PROMPTAD_MODE=export bash scripts/run_promptad_raw.sh
```

| Result | **Blocker** |
|--------|----------------|
| Exit **1** | `PROMPTAD_OUTPUT_ROOT` unset (`:?` guard). No on-disk tree was supplied for this validation. |

**Not run** with a guessed `PROMPTAD_OUTPUT_ROOT`: see **§9** (PromptAD main-line search, 2026-05-12 follow-up): **no** `CLS-*-per_sample.csv` files were found in-repo or shallow `$HOME` search, so export could not be executed.

**Expected artifacts after a successful export** (not present after this run):

- `outputs/cached_results/raw_scores/promptad/unified_raw_scores_wide.csv`
- `outputs/cached_results/raw_scores/promptad/unified_raw_scores_long.csv`
- `outputs/cached_results/raw_scores/promptad/manifest.json`

### 3.2 PaDiM (`scripts/run_padim_raw.sh`)

**Not executed.** Requires real `PADIM_DATA_ROOT`, GPU/time budget, and explicit class/seed configuration. Recorded as **not attempted** → **blocker for “reproduced from raw”** on this pass, not a script failure.

### 3.3 PatchCore (`scripts/run_patchcore_raw.sh`)

**Not executed.** Requires `PATCHCORE_DATA_ROOT`, `PATCHCORE_MODELS_RUN`, and scoring stack. Recorded as **not attempted** → **blocker for PatchCore raw path** on this pass.

---

## 4. Phase 2 — section reproduction (`scripts/reproduce_*.sh`)

All commands from repo root **after** `clean_outputs.sh`, in the order executed for this report:

| # | Command | Exit | Raw evidence used? | Mode | Notable outputs |
|---|---------|------|--------------------|------|------------------|
| 1 | `bash scripts/reproduce_sec3_promptad.sh` | **1** | No (unified raw missing; stub env unset) | **blocker** for Sec 3.1.1 empirical path | None under `outputs/cached_results/sec3_promptad/` |
| 2 | `bash scripts/reproduce_sec3_padim.sh` | **0** | No | **stub / cached-only** | `outputs/cached_results/sec3_padim/marginal_stub.csv`, `outputs/figures/sec3_padim/fastpath_done.txt` |
| 3 | `bash scripts/reproduce_sec4_systematic.sh` | **0** | No (no PromptAD unified raw → stub branch) | **stub only** | `outputs/figures/sec4_systematic/fastpath_done.txt`, `outputs/tables/sec4_systematic/fastpath_done.txt` |
| 4 | `bash scripts/reproduce_app_patchcore_tta.sh` | **0** | No PatchCore unified raw | **cached-only** (copy from `result_analysis/patchcore_tta/`) | `outputs/cached_results/app_patchcore_tta/*.csv`, figures under `outputs/figures/app_patchcore_tta/` |
| 5 | `bash scripts/reproduce_appendix.sh` | **0** | No | **cached / stub** | Appendix C/E markers; F as row 4; G stub CSV+figures |
| 6 | `bash scripts/reproduce_main.sh` | **1** | No | **blocker** at first step (same as row 1) | Stops at Sec 3.1.1 PromptAD |

**Alignment with `docs/FIGURE_MAP.md`:** With no PromptAD unified raw, the map’s **stub / fast-path** descriptions match what was produced. The **pairwise Sec 3 / Sec 4 from raw** branches were **not** exercised in this run.

---

## 5. Main-line acceptance (PromptAD chain)

Checked after Phase 2:

| Artifact | Status |
|----------|--------|
| `outputs/cached_results/sec3_promptad/pairwise_metrics.csv` | **Missing** |
| `outputs/cached_results/sec3_promptad/setting_level_metrics.csv` | **Missing** |
| `outputs/cached_results/sec3_promptad/sample_level_metrics.csv` | **Missing** |
| `outputs/cached_results/sec3_promptad/risk_coverage.csv` | **Missing** |
| `outputs/cached_results/sec4_systematic/controlled_margin_analysis.csv` | **Missing** (stub path did not write these) |
| `outputs/cached_results/sec4_systematic/failure_conditioned_signal_comparison.csv` | **Missing** |
| `outputs/cached_results/sec4_systematic/instability_regime_failure_rate.csv` | **Missing** |
| `outputs/cached_results/sec4_systematic/decision_consequence_delta_risk.csv` | **Missing** |
| `outputs/figures/sec3_promptad/*.png` | **None** (directory may be empty) |
| `outputs/figures/sec4_systematic/` | Only **`fastpath_done.txt`** (stub marker) |

**Conclusion:** The **raw_scores/promptad → sec3_promptad → sec4_systematic** empirical pipeline was **not** validated on this machine in this pass because **Stage 1 export was not run** and **no pre-existing unified raw** was placed under `outputs/`.

---

## 6. Large files and commit risk

Commands:

```bash
find outputs -type f -size +20M
find . -type f \( -name "*.pt" -o -name "*.pth" -o -name "*.pkl" -o -name "*.npy" -o -name "*.npz" -o -name "*.zip" \) | head -100
git status --short
git add -n . | head -200
```

| Check | Result |
|--------|--------|
| `outputs/` files **> 20 MiB** | **None** after this run |
| Weight-like / archive names under repo (first 100) | `./paper_figures.zip`, `./paper_figures/promptad/paper_figures.zip` (outside `outputs/`; pre-existing) |
| `git add -n .` | Staged paths were **source + docs + scripts** only; **no** `outputs/` entries in the first 200 lines of the dry-run |

**Do not commit:** anything under `outputs/` except intentional small markers if ever tracked; large `paper_figures*.zip`; any future `*.pt` / unified raw dumps.

---

## 7. Section status summary (this run)

| Section / pipeline | Status |
|--------------------|--------|
| Sec 3.1.1 PromptAD from unified raw | **blocker** (no raw; `reproduce_sec3_promptad.sh` exit 1) |
| Sec 3.1.1 PromptAD stub (`SEC3_PROMPTAD_ALLOW_STUB=1`) | **not run** in this pass |
| Sec 3.1.2 PaDiM | **stub only** |
| Sec 4 systematic from unified raw | **not exercised** (no raw) |
| Sec 4 systematic stub | **stub only** |
| Appendix F PatchCore TTA | **cached-only** |
| Appendix G signal comparison | **stub** (bundled samples) |
| `reproduce_main.sh` | **failed** (exit 1) because Sec 3.1.1 failed first |

---

## 8. Open items and next steps (factual)

1. **Provide `PROMPTAD_OUTPUT_ROOT`** pointing at an existing tree with `CLS-*-per_sample.csv`, then run  
   `FULL_RUN=1 PROMPTAD_MODE=export PROMPTAD_OUTPUT_ROOT=... bash scripts/run_promptad_raw.sh`  
   and re-run `reproduce_sec3_promptad.sh` and `reproduce_sec4_systematic.sh` to validate the **pairwise** Sec 3 / Sec 4 CSV + figure outputs.
2. **Optional:** `SEC3_PROMPTAD_ALLOW_STUB=1 bash scripts/reproduce_sec3_promptad.sh` to get exit 0 for CI without claiming raw reproduction (still **stub**).
3. **`reproduce_main.sh`:** today it fails if Sec 3.1.1 fails; orchestration is honest but **blocks** “all green” without raw or stub env.
4. **PaDiM / PatchCore Stage 1:** run only when data and model roots are known; record a new validation row when done.
5. **Pair sampling (large runs):** set `SEC3_PROMPTAD_MAX_PAIRS_PER_SETTING` / `SEC4_MAX_PAIRS_PER_SETTING` (and optional `SEC3_PROMPTAD_PAIR_SAMPLING_SEED`, `SEC4_PAIR_SAMPLING_SEED`) before `bash src/experiments/sec3_promptad_observation/run.sh` and `sec4_systematic_validation/run.sh`; see **§9.5**. **Minimal full-path success:** see **§11**.

---

## 9. PromptAD main-line search and export (2026-05-12 follow-up)

### 9.1 Search scope and rules

- **Included:** `external/PromptAD/`, `result_analysis/`, `outputs/`, repo root `result_*` directories, and shallow `find "$HOME" -maxdepth 5`** excluding** paths whose name contains `datasets`.
- **Export scanner** (`promptad_export_unified_raw.py`): only files whose basename matches  
  `CLS-<dataset>-<category>-k<shot>-seed<seed>-per_sample.csv`  
  (and not `*per_sample_instability*` / `*fusion*`).

### 9.2 Results

| Pattern | Count / note |
|---------|----------------|
| `CLS-*-per_sample.csv` under the repo (incl. `external/PromptAD`, `result_analysis`, `outputs`, depth-limited tree) | **0** |
| `*-per_sample.csv` (other names) | **1** — `src/experiments/app_signal_comparison/samples/fastpath/promptad_stub/.../ai-per_sample.csv` (toy stub; **wrong basename**, ignored by export) |
| Shallow `$HOME` search for `CLS-*-per_sample.csv` (depth 5, stderr discarded) | **0** |

### 9.3 `PROMPTAD_OUTPUT_ROOT` choice

**Not applicable:** no candidate tree containing exportable `CLS-*-per_sample.csv` files was found, so **`scripts/run_promptad_raw.sh` was not run** in this follow-up (running it would require inventing a path).

### 9.4 Downstream reproduction (Sec 3.1.1 → Sec 4 → `reproduce_main.sh`)

**Not executed** in this follow-up because unified raw was not produced. With real export, expected commands:

```bash
bash scripts/reproduce_sec3_promptad.sh
bash scripts/reproduce_sec4_systematic.sh
bash scripts/reproduce_main.sh
```

### 9.5 Pair-count limits (env → `run.sh`)

To avoid OOM / long runtimes when `n_anomaly * n_normal` per setting is huge, `run.sh` passes through:

| Variable | Passed to |
|----------|-----------|
| `SEC3_PROMPTAD_MAX_PAIRS_PER_SETTING` | `analyze_sec3_promptad_from_raw.py --max-pairs-per-setting` |
| `SEC3_PROMPTAD_PAIR_SAMPLING_SEED` | `--pair-sampling-seed` |
| `SEC4_MAX_PAIRS_PER_SETTING` | `analyze_sec4_promptad_from_raw.py --max-pairs-per-setting` |
| `SEC4_PAIR_SAMPLING_SEED` | `--pair-sampling-seed` |

Sampling strategy is recorded in `outputs/cached_results/sec3_promptad/sec3_promptad_from_raw_summary.json` and `outputs/cached_results/sec4_systematic/sec4_systematic_from_raw_summary.json` when subsampling is used.

### 9.6 Blocker — what you must provide manually

Provide an **absolute path** to a directory tree that **already contains** PromptAD-style exports, for example:

```text
<YOUR_ROOT>/.../csv/CLS-mvtec-bottle-k1-seed111-per_sample.csv
```

Then run (example):

```bash
FULL_RUN=1 PROMPTAD_MODE=export PROMPTAD_OUTPUT_ROOT="<YOUR_ROOT>" bash scripts/run_promptad_raw.sh
```

After `unified_raw_scores_wide.csv` (or `_long.csv`) appears under `outputs/cached_results/raw_scores/promptad/`, re-run the three reproduce commands above. Optionally export caps first, e.g.  
`export SEC3_PROMPTAD_MAX_PAIRS_PER_SETTING=200000`  
`export SEC4_MAX_PAIRS_PER_SETTING=200000`.

---

## 10. Suggested commit scope (documentation + code only)

Safe to commit (from `git status` / `git add -n` at validation time): modified and new **docs**, **scripts**, **`src/`** analyzers and adapters — **not** `outputs/`, not large zips.

**Should not commit:** `outputs/**`, large archives, checkpoints, or generated unified raw CSVs unless the project explicitly versions small fixtures.

---

## 11. PromptAD minimal full-path closed loop (2026-05-12)

**Status label:** **`minimal full-path verified`** — **not** a **full 81-setting** (or paper-wide multi-class) reproduction.

### 11.1 Configuration (exact)

| Field | Value |
|--------|--------|
| `PROMPTAD_MODE` | `train,infer,export` |
| `PROMPTAD_OUTPUT_ROOT` | `/home/zju/mywork/NeurIPS2026/outputs/promptad_fullpath_minimal` |
| `PROMPTAD_DATA_ROOT` | `/home/zju/datasets/mvtec` (validated; upstream reads `~/datasets/mvtec` — **same tree** on this host, **no symlink** needed) |
| `PROMPTAD_DATASETS` | `mvtec` |
| `PROMPTAD_CLASSES` | `bottle` |
| `PROMPTAD_SHOTS` | `1` |
| `PROMPTAD_SEEDS` | `111,222` |
| `PROMPTAD_GPU` | `0` |
| Short train | `PROMPTAD_TRAIN_EXTRA_ARGS="--Epoch 3 --batch-size 8 --eval-freq 1 --num-workers 2"` |

**Script fix (same day):** `PROMPTAD_EXTRA_ARGS` is no longer passed to `test_cls.py` (it rejected `--Epoch` / `--eval-freq`). Use **`PROMPTAD_TRAIN_EXTRA_ARGS`** for train-only flags; optional **`PROMPTAD_INFER_EXTRA_ARGS`** for infer. **`PROMPTAD_EXTRA_ARGS`** remains a **train-only** backward-compatible alias when `PROMPTAD_TRAIN_EXTRA_ARGS` is unset (`scripts/run_promptad_raw.sh`).

### 11.2 Stage 1 — results

| Check | Outcome |
|--------|---------|
| `find outputs/promptad_fullpath_minimal -name 'CLS-*-per_sample.csv'` | **2** files under `mvtec/k_1/csv/` (`seed111`, `seed222`) |
| `unified_raw_scores_wide.csv` | **Yes** (`outputs/cached_results/raw_scores/promptad/`; exporter logged **166** wide rows) |
| `unified_raw_scores_long.csv` | **Yes** (**498** long rows) |
| `manifest.json` | **Yes** |
| `run_promptad_raw.sh` exit | **0** |

### 11.3 Stage 2 — pairwise Sec 3 / Sec 4 + `reproduce_main.sh`

```bash
export SEC3_PROMPTAD_MAX_PAIRS_PER_SETTING=200000
export SEC4_MAX_PAIRS_PER_SETTING=200000
```

| Command | Exit |
|---------|------|
| `bash scripts/reproduce_sec3_promptad.sh` | **0** |
| `bash scripts/reproduce_sec4_systematic.sh` | **0** |
| `bash scripts/reproduce_main.sh` | **0** |

**Counts:** `pairwise_metrics.csv` **2520** rows; `setting_level_metrics.csv` **2** rows; `same_auroc_instability_pairs.csv` **0** (two seeds only; near-AUROC filter may yield zero flagged pairs).

### 11.4 Outputs (gitignored; do not commit)

- `outputs/cached_results/raw_scores/promptad/*`
- `outputs/cached_results/sec3_promptad/*`, `outputs/figures/sec3_promptad/*.png`
- `outputs/cached_results/sec4_systematic/*`, `outputs/figures/sec4_systematic/*.png`
- `outputs/promptad_fullpath_minimal/` (added to `.gitignore`)

### 11.5 Scope honesty

- **Verified:** PromptAD **train → infer → export → Sec 3.1.1 pairwise → Sec 4 pairwise → `reproduce_main.sh`** for **mvtec / bottle / k=1 / seeds 111,222** with short epochs.
- **Not verified:** full **81-setting** paper grid, VisA, all shots/seeds, or `Epoch=100` training parity with published tables.
