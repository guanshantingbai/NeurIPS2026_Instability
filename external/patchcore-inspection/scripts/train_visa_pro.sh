#!/usr/bin/env bash
# PatchCore on VisA (pro layout at VISA_ROOT). ImageNet backbone loads from torch hub on first use.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-/home/zju/miniconda3/envs/myenv/bin/python}"

VISA_ROOT="${VISA_ROOT:-$HOME/datasets/pro_visa}"
GPU="${GPU:-0}"
SEED="${SEED:-0}"
LOG_GROUP="${LOG_GROUP:-visa_IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S${SEED}}"

classes=(candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum)
dataset_flags=()
for c in "${classes[@]}"; do
  dataset_flags+=(-d "$c")
done

# CPU FAISS by default (myenv uses faiss-cpu). Add --faiss_on_gpu if you use faiss-gpu.
"$PYTHON_BIN" bin/run_patchcore.py --gpu "$GPU" --seed "$SEED" --save_patchcore_model \
  --log_group "$LOG_GROUP" --log_project VisA_PatchCore results \
  patch_core -b wideresnet50 -le layer2 -le layer3 \
  --pretrain_embed_dimension 1024 --target_embed_dimension 1024 \
  --anomaly_scorer_num_nn 1 --patchsize 3 \
  sampler -p 0.1 approx_greedy_coreset \
  dataset --resize 256 --imagesize 224 "${dataset_flags[@]}" visa "$VISA_ROOT"
