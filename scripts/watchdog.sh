#!/bin/bash
# ===========================================================================
# AFP Experiment Watchdog — runs every N seconds, logs status, detects stalls.
# Usage: nohup bash scripts/watchdog.sh &
# ===========================================================================
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

INTERVAL=${1:-300}  # default 5min
WATCHDOG_LOG="experiments/watchdog.log"
STALL_THRESHOLD=1200  # 20min no progress = stall

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$WATCHDOG_LOG"; }

log "Watchdog started (interval=${INTERVAL}s, stall_threshold=${STALL_THRESHOLD}s)"

LAST_LOG_SIZE=0
LAST_CHECK_TIME=$(date +%s)

while true; do
    sleep "$INTERVAL"
    NOW=$(date +%s)
    log "=== check $(date) ==="

    # Check master process
    MASTER_PID=$(cat experiments/master_pid.txt 2>/dev/null)
    if [ -n "$MASTER_PID" ] && kill -0 "$MASTER_PID" 2>/dev/null; then
        ELAPSED=$(ps -p "$MASTER_PID" -o etimes= 2>/dev/null | tr -d ' ')
        log "master PID=$MASTER_PID alive (${ELAPSED}s)"
    else
        log "MASTER_DEAD — checking if experiments completed normally"
        if grep -q "^done:" experiments/phase0_status.txt 2>/dev/null; then
            log "STATUS: experiments completed"
            exit 0
        elif grep -q "^ivn_fail\|^fivn_fail" experiments/phase0_status.txt 2>/dev/null; then
            log "STATUS: experiment failure detected"
        else
            log "WARNING: master died without status marker"
        fi
    fi

    # Check python experiment processes
    PY_PROCS=$(ps aux | grep -E "python.*run_(ivn|fivn)_phase0" | grep -v grep | wc -l)
    log "experiment python processes: $PY_PROCS"

    if [ "$PY_PROCS" -gt 0 ]; then
        ps aux | grep -E "python.*run_(ivn|fivn)_phase0" | grep -v grep | while read -r line; do
            PID=$(echo "$line" | awk '{print $2}')
            RSS=$(echo "$line" | awk '{print $6}')
            CPU=$(echo "$line" | awk '{print $3}')
            log "  PID=$PID CPU=${CPU}% RSS=${RSS}KB"
        done
    fi

    # GPU
    GPU_INFO=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null)
    log "GPU: $GPU_INFO"

    # Log file sizes
    IVN_LOG=$(ls -t experiments/phase0_ivn/logs/run_*.log 2>/dev/null | head -1)
    FIVN_LOG=$(ls -t experiments/phase0_fivn/logs/run_*.log 2>/dev/null | head -1)
    if [ -n "$IVN_LOG" ]; then
        IVN_SIZE=$(wc -c < "$IVN_LOG" 2>/dev/null || echo 0)
        log "IVN log: ${IVN_SIZE}B"
    fi
    if [ -n "$FIVN_LOG" ]; then
        FIVN_SIZE=$(wc -c < "$FIVN_LOG" 2>/dev/null || echo 0)
        log "FIVN log: ${FIVN_SIZE}B"
    fi

    # Stage
    if [ -f experiments/phase0_status.txt ]; then
        LAST_STAGE=$(tail -1 experiments/phase0_status.txt)
        log "stage: $LAST_STAGE"
    fi

    # Stall detection
    TOTAL_LOG_SIZE=0
    [ -n "$IVN_LOG" ] && TOTAL_LOG_SIZE=$((TOTAL_LOG_SIZE + $(wc -c < "$IVN_LOG" 2>/dev/null || echo 0)))
    [ -n "$FIVN_LOG" ] && TOTAL_LOG_SIZE=$((TOTAL_LOG_SIZE + $(wc -c < "$FIVN_LOG" 2>/dev/null || echo 0)))

    if [ "$PY_PROCS" -gt 0 ] && [ "$TOTAL_LOG_SIZE" -eq "$LAST_LOG_SIZE" ]; then
        STALL_TIME=$((NOW - LAST_CHECK_TIME))
        if [ "$STALL_TIME" -gt "$STALL_THRESHOLD" ]; then
            log "STALL_DETECTED: no log growth for ${STALL_TIME}s with active python process"
        fi
    else
        LAST_LOG_SIZE=$TOTAL_LOG_SIZE
        LAST_CHECK_TIME=$NOW
    fi

    # Check if done
    if [ "$PY_PROCS" -eq 0 ] && grep -q "^done:" experiments/phase0_status.txt 2>/dev/null; then
        log "ALL EXPERIMENTS COMPLETE — watchdog exiting"
        exit 0
    fi
done
