#!/usr/bin/env bash
# Finish panel (a) after OOM: one VisA category × WR50 per process (fresh Python heap).
# Prerequisite: checkpoint CSV with all other rows (42+), same --csv_path.
#
# If the process dies with "已杀死" / exit 137, that is usually the Linux OOM killer (SIGKILL),
# NOT a Python exception. It is often *host RAM* or cgroup limit, not "CPU usage" per se.
# Diagnose after a kill:
#   sudo dmesg -T | tail -30 | grep -iE 'oom|killed process|Out of memory'
#   free -h
#   nvidia-smi
# Isolate GPU vs full pipeline:
#   python -u padim_build_panel_a.py --diag_wr50_warmup
# Optional verbose steps during real run:
#   PADIM_MEM_DIAG=1 python -u padim_build_panel_a.py ...
#
# Env overrides:
#   CPU_THREADS=1        BLAS/OpenMP + torch CPU threads (default 1)
#   MAX_TRAIN_IMAGES=128 cap training images for Gaussian (default 128)
#   BATCH_SIZE=2         forward batch for WR50 (default 2; try 1 if still killed)
#   PADIM_STREAM_TEST_THRESHOLD=350  (in padim_protocol_b) test set >= this uses 2-pass streaming
#                                    to avoid ~30GB+ CPU RAM from torch.cat of all test features (VisA).
set -euo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
CSV="${CSV:-$REPO/result_analysis/figures/padim_panel_a_mvtec_visa_r18_wr50.csv}"
MVTEC="${MVTEC:-$HOME/datasets/mvtec}"
VISA="${VISA:-$HOME/datasets/pro_visa}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
# Reduce GPU allocator fragmentation (PyTorch 2.x)
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Limit BLAS/OpenMP threads *before* Python starts (critical for np.linalg / np.cov RAM).
CPU_THREADS="${CPU_THREADS:-1}"
export OMP_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"
export OPENBLAS_NUM_THREADS="$CPU_THREADS"
export NUMEXPR_NUM_THREADS="$CPU_THREADS"
export VECLIB_MAXIMUM_THREADS="$CPU_THREADS"

MAX_TRAIN="${MAX_TRAIN_IMAGES:-128}"
BATCH="${BATCH_SIZE:-2}"

for c in candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum; do
  echo "========== visa / wide_resnet50_2 / $c =========="
  python -u "$REPO/padim_build_panel_a.py" \
    --visa_only \
    --single_dataset visa \
    --single_backbone wide_resnet50_2 \
    --single_category "$c" \
    --mvtec_path "$MVTEC" \
    --visa_path "$VISA" \
    --csv_path "$CSV" \
    --num_workers 0 \
    --batch_size "$BATCH" \
    --max_train_images "$MAX_TRAIN" \
    --cpu_threads "$CPU_THREADS"
done

echo "Done. CSV: $CSV"
