#!/bin/bash
# Pre-download models for Phase 0 experiments via HF mirror
# Usage: nohup bash download_models.sh >> experiments/model_download.log 2>&1 &

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_PYTHON="/home/jiayu/FCL-PRM-cdspi/venv/bin/python3"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$PROJECT_DIR/experiments/model_download_${TIMESTAMP}.log"

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1

MODELS=(
    "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    "Qwen/Qwen2.5-Math-1.5B-Instruct"
    "Qwen/Qwen2.5-1.5B"
    "EleutherAI/pythia-1.4b"
)

echo "=== Model Download: $(date) ===" | tee -a "$LOG_FILE"
echo "HF_ENDPOINT=$HF_ENDPOINT" | tee -a "$LOG_FILE"

for model in "${MODELS[@]}"; do
    echo "" | tee -a "$LOG_FILE"
    echo "--- Downloading: $model ---" | tee -a "$LOG_FILE"
    START=$(date +%s)
    $VENV_PYTHON -c "
import os, sys, time
os.environ['HF_ENDPOINT'] = '$HF_ENDPOINT'
from huggingface_hub import snapshot_download
print(f'starting {sys.argv[1]}', flush=True)
t0 = time.time()
path = snapshot_download(sys.argv[1], resume_download=True)
elapsed = time.time() - t0
print(f'done: {path} ({elapsed:.0f}s)', flush=True)
" "$model" 2>&1 | tee -a "$LOG_FILE"
    END=$(date +%s)
    echo "  elapsed: $(( (END-START)/60 ))m" | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"
echo "=== All models downloaded: $(date) ===" | tee -a "$LOG_FILE"
