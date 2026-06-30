#!/bin/bash
# ===========================================================================
# AFP Phase 0 Master Experiment Launcher
# Runs IVN (weight-space) then F-IVN (function-space) experiments.
#
# Usage: nohup bash scripts/run_phase0_master.sh &
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_PYTHON="/home/jiayu/FCL-PRM-cdspi/venv/bin/python3"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_DIR/experiments"
MASTER_LOG="$LOG_DIR/master_${TIMESTAMP}.log"
STATUS_FILE="$LOG_DIR/phase0_status.txt"

# HF mirror for fast downloads (no VPN needed)
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTHONUNBUFFERED=1

# ==============================
log_msg() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "$MASTER_LOG"
}

fail() {
    log_msg "FATAL: $*"
    echo "FAILED:$(date):$*" >> "$STATUS_FILE"
    exit 1
}

mark_stage() {
    echo "$1:$(date)" >> "$STATUS_FILE"
}

# ==============================
log_msg "========================================="
log_msg "AFP Phase 0 Master Launcher"
log_msg "HF_ENDPOINT=$HF_ENDPOINT"
log_msg "Python: $($VENV_PYTHON --version)"
log_msg "========================================="

# ---- Check venv ----
$VENV_PYTHON -c "import torch; print(f'[sanity] torch {torch.__version__}, cuda={torch.cuda.is_available()}')" 2>&1 | tee -a "$MASTER_LOG" \
    || fail "torch import failed"
mark_stage "sanity_pass"

# ==============================
# Experiment A: IVN (weight-space)
# ==============================
IVN_LOG="$LOG_DIR/phase0_ivn/logs/run_${TIMESTAMP}.log"
log_msg ""
log_msg "=== Experiment A: IVN (weight-space) ==="
log_msg "  log: $IVN_LOG"

mark_stage "ivn_start"

cd "$PROJECT_DIR"
$VENV_PYTHON scripts/run_ivn_phase0.py \
    --importance mas \
    --gate rational \
    --tau 0.5 \
    --max-rounds 30 \
    > "$IVN_LOG" 2>&1
IVN_EXIT=$?

if [ $IVN_EXIT -eq 0 ]; then
    log_msg "IVN completed successfully (exit=$IVN_EXIT)"
    mark_stage "ivn_pass"
else
    log_msg "IVN FAILED (exit=$IVN_EXIT) — check $IVN_LOG and crashes/"
    mark_stage "ivn_fail"
    # Continue with F-IVN anyway — independent experiment
fi

# ==============================
# Experiment B: F-IVN (function-space)
# ==============================
FIVN_LOG="$LOG_DIR/phase0_fivn/logs/run_${TIMESTAMP}.log"
log_msg ""
log_msg "=== Experiment B: F-IVN (function-space) ==="
log_msg "  log: $FIVN_LOG"

mark_stage "fivn_start"

cd "$PROJECT_DIR"
$VENV_PYTHON scripts/run_fivn_phase0.py \
    --tau 0.5 \
    --max-rounds 30 \
    > "$FIVN_LOG" 2>&1
FIVN_EXIT=$?

if [ $FIVN_EXIT -eq 0 ]; then
    log_msg "F-IVN completed successfully (exit=$FIVN_EXIT)"
    mark_stage "fivn_pass"
else
    log_msg "F-IVN FAILED (exit=$FIVN_EXIT) — check $FIVN_LOG"
    mark_stage "fivn_fail"
fi

# ==============================
# Summary
# ==============================
log_msg ""
log_msg "========================================="
log_msg "Phase 0 complete"
log_msg "  IVN:   exit=$IVN_EXIT   log=$IVN_LOG"
log_msg "  F-IVN: exit=$FIVN_EXIT  log=$FIVN_LOG"
log_msg "========================================="
mark_stage "done"

echo "done:$(date)" >> "$STATUS_FILE"
