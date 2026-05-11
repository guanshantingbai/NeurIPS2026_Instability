# External Nested Repos Audit

审计时间：2026-05-11  
审计范围：
- `external/PromptAD`
- `external/patchcore-inspection`
- `external/PaDiM-Anomaly-Detection-Localization-master`

已执行命令（每个仓库）：
- `git status --short`
- `git branch --show-current`
- `git remote -v`
- `git log -1 --oneline`
- `git diff --stat`
- `git diff --name-only`
- `git ls-files --others --exclude-standard`

> 说明：三者均是独立嵌套 git 仓库（保留 `.git`，本次仅审计，不删除/不移动/不提交）。

## 1) external/PromptAD

- **是否是 git repo**：是
- **当前分支**：`master`
- **当前 commit**：`0f86ce0 Update train_cls.py`
- **remote**：`https://github.com/FuNz-0/PromptAD`（fetch/push）
- **tracked modified（`git diff --name-only`）**：13 个
  - `README.md`
  - `datasets/__init__.py`
  - `datasets/dataset.py`
  - `datasets/mvtec.py`
  - `datasets/visa.py`
  - `run_cls.py`, `run_seg.py`
  - `test_cls.py`, `test_seg.py`
  - `train_cls.py`, `train_seg.py`
  - `utils/metrics.py`
  - `utils/training_utils.py`
- **untracked（`git ls-files --others --exclude-standard`）**：9983 个（数量极大）
  - 主要类别：`__pycache__/`、`logs/`、`result_*`、`result_analysis/`、`pairwise_penalty_analysis_out/`
  - 同时包含大量新增脚本（如 `utils/seed_killer_evidence_pipeline.py`、`utils/promptad_strengthening_experiments.py`、`utils/supplementary_signal_baselines.py` 等）

### 与本论文复现相关的必要修改（推断）

优先级较高（章节复现/分析链路直接相关）：
- `utils/seed_killer_evidence_pipeline.py`
- `utils/promptad_strengthening_experiments.py`
- `utils/mechanism_driven_analysis.py`
- `utils/supplementary_signal_baselines.py`
- `utils/plot_mechanism_combined_section4.py`
- `utils/plot_failure_merged_section4.py`
- `utils/build_paper_figures_section4.py`
- `bash/run_*all_settings.sh`（批量执行脚本）

### 应忽略（实验产物/缓存/日志）

- `__pycache__/`
- `logs/`
- `result_offline_*`, `result_round1/`, `result_seed_search/`
- `result_analysis/` 下大体量中间产物（除轻量必要表格外）

---

## 2) external/patchcore-inspection

- **是否是 git repo**：是
- **当前分支**：`main`
- **当前 commit**：`fcaa92f Remove wrong symbolic link`
- **remote**：`https://github.com/amazon-science/patchcore-inspection`（fetch/push）
- **tracked modified（`git diff --name-only`）**：3 个
  - `bin/load_and_evaluate_patchcore.py`
  - `bin/run_patchcore.py`
  - `src/patchcore/backbones.py`
- **untracked（`git ls-files --others --exclude-standard`）**：31 个
  - 关键条目：
    - `scripts/run_patchcore_tta_mechanism.py`
    - `src/patchcore/datasets/visa.py`
    - `scripts/train_visa_pro.sh`
    - `scripts/download_mvtec_patchcore_lfs.sh`
  - 产物条目：
    - `logs/visa_patchcore_train.*`
    - `results/VisA_PatchCore/...`（模型和结果）

### 与本论文复现相关的必要修改（推断）

- `scripts/run_patchcore_tta_mechanism.py`（Appendix F 核心）
- `src/patchcore/datasets/visa.py`（VisA 数据支持）
- `bin/run_patchcore.py`、`bin/load_and_evaluate_patchcore.py`、`src/patchcore/backbones.py`（训练/评估入口与模型配置适配）

### 应忽略（实验产物/缓存/日志）

- `logs/`
- `results/VisA_PatchCore/...`（权重与完整结果）

---

## 3) external/PaDiM-Anomaly-Detection-Localization-master

- **是否是 git repo**：是
- **当前分支**：`main`
- **当前 commit**：`616004b Create LICENSE`
- **remote**：`https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master.git`（fetch/push）
- **tracked modified（`git diff --name-only`）**：2 个
  - `datasets/mvtec.py`
  - `main.py`
- **untracked（`git ls-files --others --exclude-standard`）**：5027 个（数量极大）
  - 关键脚本：
    - `padim_seed_killer_evidence_pipeline.py`
    - `run_padim_seed_killer_one_click.sh`
    - `padim_protocol_b_one_run.py`
    - `padim_protocol_b_mvtec_multiclass.py`
    - `padim_instability_protocol_ab.py`
    - `build_padim_paper_figures_section4.py`
    - `merge_padim_fig34_fig56_section4.py`
  - 大量产物目录：
    - `mvtec_result_full_wr50/`
    - `padim_result_seed_search*/`
    - `protocol_*/`
    - `visa_result_*/`
    - `figures/`, `result_analysis/`

### 与本论文复现相关的必要修改（推断）

- `padim_seed_killer_evidence_pipeline.py`
- `run_padim_seed_killer_one_click.sh`
- `padim_protocol_b_one_run.py`
- `padim_instability_protocol_ab.py`
- `build_padim_paper_figures_section4.py`
- `merge_padim_fig34_fig56_section4.py`
- `datasets/visa.py`（如涉及 VisA 路线）

### 应忽略（实验产物/缓存/日志）

- `mvtec_result_full_wr50/`（超大）
- `padim_result_seed_search*/`
- `protocol_*/`
- `visa_result_*/`
- `figures/` 下大图缓存、`result_analysis/` 下重产物、`__pycache__/`

---

## 结论与建议（仅审计结论）

- 三个 external 仓库当前都处于**强 dirty 状态**，且 untracked 规模很大（PromptAD 9983、PaDiM 5027）。
- 论文复现所需代码和实验产物高度混杂，尤其 PromptAD/PaDiM。
- 后续若要可控版本管理，建议在 external 仓库内至少区分两类：
  1) **必要复现脚本改动**（保留）
  2) **结果/缓存/日志**（统一忽略）

本文件仅记录审计，不执行任何清理动作。
