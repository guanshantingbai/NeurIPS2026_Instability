# External Vendoring Strategy

本文档记录将 `external/` 下三个嵌套 git 仓库转为**外层仓库管理的 vendored 源码**的处理策略。  
前置审计见 `docs/EXTERNAL_AUDIT.md`（保留不删）。

## 目标

- 保留论文复现所需的 **Python 源码、shell 脚本、轻量配置**。
- **物理删除**（或等价清空）实验产物、缓存、日志、权重、大结果目录，降低体积与误提交风险。
- **删除** `external/*/.git`，消除嵌套仓库，便于外层 `git add external/...` 跟踪源码。
- **不**在本步骤自动 `git commit`（由维护者审阅后再提交）。

## 通用原则

1. **源码优先**：`*.py`、`*.sh`、小型 `*.md` / `*.txt` / `*.yaml` / `*.json` 配置保留。
2. **产物外置**：`result_*`、`result_analysis/`、`logs/`、`__pycache__/`、预训练与索引权重（`*.pth`、`*.pt`、`*.pkl`、`*.faiss` 等）删除或依赖 `.gitignore` 兜底。
3. **与上游对齐**：保留的脚本尽量为论文分析链实际调用的入口；其余大目录按审计结论删除。

## external/PromptAD

**保留**

- 上游原有 Python 包与入口：`PromptAD/`、`datasets/`、`run_*.py`、`train_*.py`、`test_*.py` 等已跟踪源码。
- 已修改的 tracked 文件（见审计中的 13 个路径）。
- 论文复现相关 **新增** `utils/*.py`（如 seed-killer、strengthening、supplementary 等）。
- `bash/run_*.sh` 批量脚本。

**删除（物理）**

- `__pycache__/`（全树）
- `logs/`
- `result_offline_*`、`result_round1/`、`result_seed_search/`
- `result_analysis/`（论文分析中间产物，体积大）
- `pairwise_penalty_analysis_out/`
- 散落的权重文件：`*.pth`、`*.pt` 等（若仍存在）

**说明**

- `appendix_promptad_minimal/` 若体积较小可保留；若与 `result_analysis` 同属一次性出图缓存，可按需删除（本次以删除大结果目录为主）。

## external/patchcore-inspection

**保留**

- `bin/*.py`
- `src/patchcore/**/*.py`
- `scripts/run_patchcore_tta_mechanism.py`
- `scripts/train_visa_pro.sh`
- `scripts/download_mvtec_patchcore_lfs.sh`
- 其余 `scripts/*.py` / `scripts/*.sh` 若为小体积工具脚本，默认保留（未列入删除清单）。

**删除（物理）**

- `logs/`
- `results/`
- `models/`（预训练 PatchCore 权重与索引，体积大）
- 全树 `__pycache__/`

## external/PaDiM-Anomaly-Detection-Localization-master

**保留**

- 原始仓库中的核心源码：`main.py`、`datasets/*.py` 等。
- 论文相关脚本：`padim_protocol_b*.py`、`padim_seed_killer_evidence_pipeline.py`、`run_padim_seed_killer_one_click.sh`、`padim_instability_protocol_ab.py`、`build_padim_paper_figures_section4.py`、`merge_padim_fig34_fig56_section4.py` 及其他体量正常的 `padim_*.py` 分析脚本。

**删除（物理）**

- `mvtec_result_full_wr50/`
- `padim_result_seed_search*/`
- `protocol_*/`
- `visa_result_*/`
- `result_analysis/`、`figures/`（大图缓存）
- 全树 `__pycache__/`
- 散落权重与大二进制：`*.pth`、`*.pt`、`*.pkl`（大文件）、`*.npy` 等（在仍存在的路径下清理）

## 根目录 `.gitignore` 补充

在 `external/` 下增加兜底规则，防止将来再次生成产物被误提交（与物理删除互补）。

## 嵌套 `.git` 删除

执行（仅元数据）：

- `rm -rf external/PromptAD/.git`
- `rm -rf external/patchcore-inspection/.git`
- `rm -rf external/PaDiM-Anomaly-Detection-Localization-master/.git`

## 验收（本步骤末尾由脚本输出）

- `git status --short external | head -200` 与行数统计
- `find external -type f -size +20M`
- `find external -type d \( -name "__pycache__" -o -name "logs" -o -name "results" \)`
- 预估将被纳入 Git 的文件数量（`git add -n` 计数）
- **不**执行 `git commit`

---

## 执行记录（2026-05-11，本机）

### 清理后体积

| 目录 | 清理后 `du -sh` |
|------|-----------------|
| `external/PromptAD` | 约 3.4M |
| `external/patchcore-inspection` | 约 2.3M |
| `external/PaDiM-Anomaly-Detection-Localization-master` | 约 2.7M |

### 嵌套 `.git`

已删除：`external/PromptAD/.git`、`external/patchcore-inspection/.git`、`external/PaDiM-Anomaly-Detection-Localization-master/.git`。

### 验收命令结果摘要

- `find external -type f -size +20M`：**无输出**（无大于 20MB 文件）。
- `find external -type d \( -name "__pycache__" -o -name "logs" -o -name "results" \)`：**无输出**（无残留目录）。
- `find external -type f | wc -l`：**275** 个文件。
- `git add -n external/ | wc -l`：**270** 行（与将纳入索引的路径数一致；与 find 计数接近，差异可能来自空目录或未被枚举的边界情况）。
- `git add -n` 抽样检查：**无** `.pth` / `.pt` / `.pkl` / `.faiss` / `.npy` / `.npz` 等权重类扩展名进入暂存预览。

### 关键论文脚本存在性（抽样）

以下路径在清理后仍存在：

- `external/patchcore-inspection/scripts/run_patchcore_tta_mechanism.py`
- `external/PromptAD/utils/seed_killer_evidence_pipeline.py`
- `external/PromptAD/utils/supplementary_signal_baselines.py`
- `external/PaDiM-Anomaly-Detection-Localization-master/padim_seed_killer_evidence_pipeline.py`
- `external/PaDiM-Anomaly-Detection-Localization-master/run_padim_seed_killer_one_click.sh`

### 关于 `git status --short external`

未暂存时，Git 可能将整棵 `external/<name>/` 显示为三行 `??`；实际可跟踪文件数以 `git add -n` / `find` 为准。
