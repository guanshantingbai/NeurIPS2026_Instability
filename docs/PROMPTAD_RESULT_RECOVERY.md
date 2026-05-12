# PromptAD per_sample result recovery (read-only audit)

**Date:** 2026-05-12  
**Scope:** Locate existing PromptAD-style `*-per_sample.csv` / `CLS-*-per_sample.csv` evidence on this machine. **No** delete, move, copy, restore, or `git clean` was executed as part of this audit.

---

## 1. Linux Trash (`~/.local/share/Trash/`)

### Commands run

```bash
find ~/.local/share/Trash -iname "*per_sample*.csv" 2>/dev/null
find ~/.local/share/Trash -iname "*PromptAD*" 2>/dev/null
find ~/.local/share/Trash -iname "*result*" 2>/dev/null
```

(Additional listing: `Trash/files` and `Trash/info` are populated mostly with screenshot PNGs and matching `.trashinfo` files.)

### Findings

| Search | Result |
|--------|--------|
| `*per_sample*.csv` under Trash | **No matches** |
| `*PromptAD*` under Trash | **No matches** |
| `*result*` under Trash | **No CSV hits** in the truncated search; Trash contains unrelated files (e.g. screenshots). Snap-related trash paths under `~/snap/code/.../Trash/files/result_RPCA` appeared in a **separate** directory-name search (§3); not inspected as PromptAD exports. |

**Conclusion:** No **`CLS-*-per_sample.csv`** (or other `*per_sample*.csv`) was found in the user Trash paths above.

---

## 2. Home-wide file search (pruned)

### Commands run

```bash
find ~ \
  -path "*/datasets/*" -prune -o \
  -path "*/.cache/*" -prune -o \
  -path "*/miniconda*/*" -prune -o \
  -iname "CLS-*-per_sample.csv" -print 2>/dev/null | head -200

find ~ \
  -path "*/datasets/*" -prune -o \
  -path "*/.cache/*" -prune -o \
  -iname "*per_sample.csv" -print 2>/dev/null | head -200
```

### Findings

| Pattern | Matches (first 200 lines) |
|---------|---------------------------|
| `CLS-*-per_sample.csv` | **None** |
| `*per_sample.csv` | **One** file (not `CLS-*`):  
  `/home/zju/mywork/NeurIPS2026/src/experiments/app_signal_comparison/samples/fastpath/promptad_stub/result_seed_search/mvtec__bottle__k1/111/ai-per_sample.csv` |

**`stat`** on that file (read-only):

```text
/home/zju/mywork/NeurIPS2026/src/experiments/app_signal_comparison/samples/fastpath/promptad_stub/result_seed_search/mvtec__bottle__k1/111/ai-per_sample.csv 491 bytes  2026-05-11 20:40:43 +0800
```

This is a **bundled stub** (`ai-per_sample.csv`); the Stage 1 exporter (`promptad_export_unified_raw.py`) expects basenames matching **`CLS-<dataset>-<class>-k<k>-seed<seed>-per_sample.csv`**, so this stub is **not** ingestible as-is.

---

## 3. Directory-name search (heuristic “history” locations)

### Command run

```bash
find ~ \
  -path "*/datasets/*" -prune -o \
  -type d \( \
    -iname "result*" -o \
    -iname "*seed*" -o \
    -iname "*promptad*" -o \
    -iname "*analysis*" \
  \) -print 2>/dev/null | head -300
```

### Notable paths (first 300; many are unrelated conda/site-packages)

**Potentially relevant (project / clone):**

- `/home/zju/codes/AD/PromptAD` and `/home/zju/codes/AD/PromptAD/result` (and subpaths like `result/phase1_cleaned/.../Seed_111`)
- `/home/zju/mywork/IJCAI2026/PromptAD` and `.../PromptAD/result`, `.../analysis`
- `/home/zju/mywork/NeurIPS2026/external/PromptAD`
- `/home/zju/mywork/NeurIPS2026/result_analysis` (tracked **PatchCore** artifacts only; see §4)
- `/home/zju/.cursor/projects/home-zju-codes-AD-PromptAD` (IDE metadata, not model outputs)

**Targeted follow-up** on standalone trees (read-only):

```bash
find /home/zju/codes/AD/PromptAD /home/zju/mywork/IJCAI2026/PromptAD \
  -name 'CLS-*-per_sample.csv' 2>/dev/null | head -80
```

**Result:** **No** `CLS-*-per_sample.csv` under those two roots in this scan.

**Historical path (from shell history, not verified on disk):**  
`~/.bash_history` contains many commands under **`/home/zju/mywork/NeurIPS2026/PromptAD/result_round1/`** (e.g. `.../csv/CLS-mvtec-bottle-k1-seed111-per_sample.csv`).  

**Current check:**

```bash
ls /home/zju/mywork/NeurIPS2026/PromptAD
```

→ **`No such file or directory`** on this machine at audit time. So the **previous** result tree is **not** present next to the current repo layout (may have been removed, renamed, or never synced to this host).

---

## 4. Git history (NeurIPS2026 repo)

### Commands run

```bash
git log --all --oneline -- "**/*per_sample*.csv"
git log --all --oneline -- "**/result_analysis/**"
git log --all --oneline -- "**/PromptAD/**"
git status --short
git log --stat --all --max-count=20
```

### Findings

| Query | Result |
|--------|--------|
| `**/*per_sample*.csv` | **One** commit touching **only** the stub:  
  `src/experiments/app_signal_comparison/samples/fastpath/promptad_stub/.../ai-per_sample.csv` (commit `1f23feb`, 2026-05-11). **No** `CLS-*-per_sample.csv` ever tracked. |
| `**/result_analysis/**` | Initial release (`bcd1ecb`) added **PatchCore TTA** CSVs/PDFs/PNGs only — **no** PromptAD per-sample CSVs. |
| `**/PromptAD/**` | Large vendor commit (`7cfb066`) for `external/PromptAD/**` source — **not** per-run `result_round1` CSV trees. |
| `git status` | Working tree shows **modified/untracked docs and scripts** from later work; nothing indicates deleted tracked `CLS-*` files (they were never in-tree). |

**Conclusion:** Git cannot recover **`CLS-*-per_sample.csv`** for this project because those artifacts were **not committed** (by design / `.gitignore`).

---

## 5. Shell history — `rm` / `clean_outputs` / results

### Command run

```bash
tail -500 ~/.bash_history | grep -E "rm |clean_outputs|result_round|PromptAD|per_sample|git clean" | tail -80
```

(`history` in non-interactive shells is often empty; **`.bash_history`** was used instead.)

### Observations

- History shows **extensive use** of **`/home/zju/mywork/NeurIPS2026/PromptAD/result_round1`** and related `python test_cls.py` / `verify_per_sample_auroc.py` commands — consistent with a **past** local result tree.
- Grep hits include generic **`rm -rf nanoGPT/.git`** (unrelated).
- **No** explicit `rm ... result_round1` or `rm ... CLS-` line appeared in the **last ~80 filtered lines** of `.bash_history` shown in the audit snippet; absence in a tail window is **not** proof files were not removed earlier or by other tools.

**`scripts/clean_outputs.sh`:** removes **`outputs/`** under the repo; it does **not** delete `~/mywork/NeurIPS2026/PromptAD/` by default. If results were only under `outputs/cached_results/raw_scores/promptad/`, a clean could remove **exported** unified CSVs, but would not explain loss of a separate **`.../PromptAD/result_round1`** tree unless outputs were the only copy.

---

## 6. Recoverable paths (this audit)

| Category | Path | Note |
|----------|------|------|
| Stub only | `.../promptad_stub/.../ai-per_sample.csv` | Present; **wrong** basename for export |
| Trash | — | **No** `*.csv` / `*.pt` / `*.pth` in XDG Trash; snap Trash had **0-byte** `consolidated.00.pth` only — see **§9** |
| Git | — | **No** `CLS-*` blobs |
| Prior user tree (history) | `/home/zju/mywork/NeurIPS2026/PromptAD/result_round1/` | **Missing** on disk now |

**Nothing listed above is a ready-to-point `PROMPTAD_OUTPUT_ROOT` for `CLS-*-per_sample.csv` export** on this host at audit time.

---

## 7. If nothing is found — minimal regeneration plan (informational only)

1. **Locate any backup** of the old tree (same machine: other disks, `rsync` targets, tarballs, another clone) matching  
   `**/csv/CLS-*-per_sample.csv`.
2. **Re-run inference** with upstream `test_cls.py` (or your batch scripts) writing to a **new** `--root-dir` / `result_round1` layout, then set  
   `PROMPTAD_OUTPUT_ROOT` to that root and run **`FULL_RUN=1 PROMPTAD_MODE=export bash scripts/run_promptad_raw.sh`** (see `docs/FULLPATH_PROMPTAD.md`).
3. **Smallest smoke test:** one setting, e.g. `mvtec` / `bottle` / `k=1` / one `seed`, confirm one file named like  
   `CLS-mvtec-bottle-k1-seed111-per_sample.csv` under a `csv/` subdirectory, then export.

---

## 9. Trash follow-up — any `*.csv` and weights `*.pth` / `*.pt` (read-only, 2026-05-12)

Additional scans requested: **all** files ending in **`.csv`**, and weight-like **`.pth`** / **`.pt`**, under common Trash locations.

### 9.1 XDG Trash (`~/.local/share/Trash/`)

```bash
find ~/.local/share/Trash -type f -iname '*.csv' 2>/dev/null
find ~/.local/share/Trash -type f \( -iname '*.pth' -o -iname '*.pt' \) 2>/dev/null
```

| Pattern | Result |
|---------|--------|
| `*.csv` | **0** files |
| `*.pth` / `*.pt` | **0** files |

(`Trash/files` top-level and full subtree counts for `*.csv` under `Trash/files` were also **0**.)

### 9.2 Snap-isolated Trash (`~/snap/**/Trash/files/`)

```bash
find ~/snap -path '*/Trash/files/*' -type f -iname '*.csv' 2>/dev/null
find ~/snap -path '*/Trash/files/*' -type f \( -iname '*.pth' -o -iname '*.pt' \) 2>/dev/null
```

| Pattern | Result |
|---------|--------|
| `*.csv` | **0** files |
| `*.pth` / `*.pt` | **2** paths (same basename in two snap revisions): |

`stat` (read-only):

```text
/home/zju/snap/code/237/.local/share/Trash/files/consolidated.00.pth | 0 bytes | 2025-04-19 20:20:19 +0800
/home/zju/snap/code/238/.local/share/Trash/files/consolidated.00.pth | 0 bytes | 2025-04-19 20:20:19 +0800
```

**Interpretation:** **empty (0-byte)** placeholder files; **not** usable model weights and **not** PromptAD-related by name/path alone.

### 9.3 Summary vs §1

Section **§1** already reported no `*per_sample*.csv` / `*PromptAD*` hits in XDG Trash. This follow-up confirms there are **no** **any**-suffix **`.csv`** files in **`~/.local/share/Trash`** either, and **no** non-empty weight checkpoints there. Snap Trash only had the two **0-byte** `.pth` entries above.

---

## 10. Explicit non-actions (compliance)

This document was produced using **read-only** inspection (`find`, `ls`, `stat`, `git log`, `grep` on history). **No** restore from Trash, **no** `cp`/`mv`/`rm`, **no** `git checkout`/`restore` of artifacts, and **no** re-run of training/inference was performed as part of generating this report.
