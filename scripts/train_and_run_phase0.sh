#!/bin/bash
# ===========================================================================
# AFP Phase 0 Auto Pipeline: Train → IVN → Report
#
# Usage:
#   nohup bash scripts/train_and_run_phase0.sh &
#   tail -f experiments/phase0_training/pipeline_*.log
#
# Process Protection Rules:
#   - All processes run via nohup + &
#   - stdout/stderr → timestamped logs in experiments/
#   - Python uses -u (unbuffered)
# ===========================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
# No expandable_segments: useful on discrete GPU with limited VRAM, but on
# GB10 unified memory (121 GB) it causes allocator fragmentation overhead.
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

VENV=/home/jiayu/FCL-PRM-cdspi/venv/bin/python3
LOG_DIR=experiments/phase0_training
mkdir -p "$LOG_DIR"

TS=$(date +%Y%m%d_%H%M%S)
PIPE_LOG="$LOG_DIR/pipeline_${TS}.log"
exec > >(tee -a "$PIPE_LOG") 2>&1

echo "=============================================="
echo " AFP Phase 0 Pipeline — $(date)"
echo "=============================================="
echo ""
echo "Steps:"
echo "  1. Pre-tokenize eval data for Pythia"
echo "  2. Train code agent (full-FT)"
echo "  3. Train medical agent (full-FT)"
echo "  4. Run IVN experiment"
echo "  5. Generate report"
echo ""

# Helper: wait for a PID and capture its exit code safely with set -e
_wait_pid() {
    local pid=$1
    local rc
    set +e
    wait "$pid"
    rc=$?
    set -e
    return $rc
}

# ---------------------------------------------------------------------------
# Step 1: Pre-tokenize eval data for Pythia (so load_data() doesn't call HF)
# ---------------------------------------------------------------------------
echo "--- [1/5] Pre-tokenize eval data for Pythia ---"

for domain in code medical; do
    CACHE="data/versaprm/versa_prm_${domain}_EleutherAI__pythia-1.4b.pt"
    if [ -f "$CACHE" ]; then
        echo "  [skip] $CACHE exists"
    else
        echo "  [prep] $CACHE ..."
        $VENV -u -c "
import sys; sys.path.insert(0, 'src')
from scripts.run_ivn_phase0 import load_data
import os; os.environ['HF_DATASETS_OFFLINE'] = '1'
data = load_data('$domain', 'EleutherAI/pythia-1.4b')
print(f'  -> {data[\"n\"]} examples cached')
" || echo "  [warn] pre-tokenize failed (will try again in IVN step)"
    fi
done
echo ""

# ---------------------------------------------------------------------------
# Step 2: Train code agent
# ---------------------------------------------------------------------------
echo "--- [2/5] Train code agent ---"
CODE_LOG="$LOG_DIR/train_code_${TS}.log"

nohup $VENV -u scripts/train_agent.py --domain code \
    >> "$CODE_LOG" 2>&1 &
CODE_PID=$!
echo "  PID=$CODE_PID, log=$CODE_LOG"

echo -n "  waiting"
while kill -0 $CODE_PID 2>/dev/null; do echo -n "."; sleep 30; done || true
_wait_pid $CODE_PID || { echo ""; echo "  [FAIL] code training exited"; tail -20 "$CODE_LOG"; exit 1; }
echo ""
echo "  [OK] code agent trained"
grep "best val_loss" "$CODE_LOG" | tail -1 || true
echo ""

# ---------------------------------------------------------------------------
# Step 3: Train medical agent
# ---------------------------------------------------------------------------
echo "--- [3/5] Train medical agent ---"
MED_LOG="$LOG_DIR/train_medical_${TS}.log"

nohup $VENV -u scripts/train_agent.py --domain medical \
    >> "$MED_LOG" 2>&1 &
MED_PID=$!
echo "  PID=$MED_PID, log=$MED_LOG"

echo -n "  waiting"
while kill -0 $MED_PID 2>/dev/null; do
    echo -n "."
    sleep 30
done
_wait_pid $MED_PID || { echo ""; echo "  [FAIL] medical training exited"; tail -20 "$MED_LOG"; exit 1; }
echo ""
echo "  [OK] medical agent trained"
grep "best val_loss" "$MED_LOG" | tail -1 || true
echo ""

# ---------------------------------------------------------------------------
# Step 4: Run IVN experiment
# ---------------------------------------------------------------------------
echo "--- [4/5] IVN experiment ---"
IVN_LOG="$LOG_DIR/ivn_${TS}.log"

nohup $VENV -u scripts/run_ivn_phase0.py \
    --teacher EleutherAI/pythia-1.4b \
    --student EleutherAI/pythia-1.4b \
    --base-model EleutherAI/pythia-1.4b \
    --teacher-domain code \
    --student-domain medical \
    --weights experiments/trained_models \
    --importance magnitude_l2 \
    --gate rational \
    --tau 0.5 \
    --max-rounds 30 \
    >> "$IVN_LOG" 2>&1 &
IVN_PID=$!
echo "  PID=$IVN_PID, log=$IVN_LOG"

echo -n "  waiting"
while kill -0 $IVN_PID 2>/dev/null; do
    echo -n "."
    sleep 30
done
_wait_pid $IVN_PID || { echo ""; echo "  [FAIL] IVN experiment exited"; tail -20 "$IVN_LOG"; exit 1; }
echo ""
echo "  [OK] IVN experiment completed"

# Show key results
if [ -f experiments/phase0_ivn/ivn_results.json ]; then
    $VENV -c "
import json
r = json.load(open('experiments/phase0_ivn/ivn_results.json'))
print(f'  importance_cosine: {r[\"importance_cosine\"]:.3f}')
print(f'  IVN net: {r[\"ivn\"][\"net\"]:+.4f}')
print(f'  AFP 1-shot net: {r[\"afp_oneshot\"][\"net\"]:+.4f}')
print(f'  FedAvg net: {r[\"fedavg\"][\"net\"]:+.4f}')
"
fi
echo ""

# ---------------------------------------------------------------------------
# Step 5: Generate report
# ---------------------------------------------------------------------------
echo "--- [5/5] Generate report ---"
if [ -f experiments/phase0_ivn/ivn_results.json ]; then
    $VENV scripts/generate_phase0_report.py >> "$LOG_DIR/report_${TS}.log" 2>&1 || true
    echo "  [OK] report generated"
else
    echo "  [warn] ivn_results.json not found, skipping report"
fi
echo ""

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo "=============================================="
echo " Pipeline complete — $(date)"
echo " Logs:    $LOG_DIR/*_${TS}.log"
echo " Results: experiments/phase0_ivn/ivn_results.json"
echo "=============================================="
