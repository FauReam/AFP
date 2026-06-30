#!/bin/bash
# ===========================================================================
# AFP Experiment Monitor — checks experiment health and progress
# Usage: bash scripts/monitor.sh [--watch]
# ===========================================================================
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "=== AFP Monitor: $(date) ==="
echo ""

# 1. Master process
MASTER_PID=$(cat experiments/master_pid.txt 2>/dev/null)
if [ -n "$MASTER_PID" ] && kill -0 "$MASTER_PID" 2>/dev/null; then
    echo "[OK] Master process PID=$MASTER_PID running"
    ps -p $MASTER_PID -o pid,pcpu,pmem,etime,cmd --no-headers 2>/dev/null
else
    echo "[!!] Master process NOT running"
fi

# 2. Stage status
if [ -f experiments/phase0_status.txt ]; then
    echo ""
    echo "=== Stages ==="
    cat experiments/phase0_status.txt
fi

# 3. Python processes
echo ""
echo "=== Python processes ==="
ps aux | grep -E "python.*(run_ivn|run_fivn|download)" | grep -v grep | while read line; do
    echo "  $line"
done

# 4. GPU
echo ""
echo "=== GPU ==="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv 2>/dev/null || echo "nvidia-smi unavailable"

# 5. IVN log
IVN_LOG=$(ls -t experiments/phase0_ivn/logs/run_*.log 2>/dev/null | head -1)
if [ -n "$IVN_LOG" ]; then
    echo ""
    echo "=== IVN Log (last 20 lines of $IVN_LOG) [$(( $(wc -l < "$IVN_LOG") )) lines] ==="
    tail -20 "$IVN_LOG" 2>/dev/null
fi

# 6. F-IVN log
FIVN_LOG=$(ls -t experiments/phase0_fivn/logs/run_*.log 2>/dev/null | head -1)
if [ -n "$FIVN_LOG" ]; then
    echo ""
    echo "=== F-IVN Log (last 20 lines of $FIVN_LOG) [$(( $(wc -l < "$FIVN_LOG") )) lines] ==="
    tail -20 "$FIVN_LOG" 2>/dev/null
fi

# 7. Model download
echo ""
echo "=== Model download ==="
tail -5 experiments/model_pre_download.log 2>/dev/null
tail -5 experiments/model_download_*.log 2>/dev/null

# 8. Crashes
echo ""
echo "=== Crashes ==="
ls experiments/phase0_ivn/crashes/ 2>/dev/null && echo "IVN crash detected!"
ls experiments/phase0_fivn/crashes/ 2>/dev/null && echo "F-IVN crash detected!"

# 9. Stall detection
echo ""
echo "=== Stall check ==="
LAST_MOD=$(stat -c %Y "$IVN_LOG" 2>/dev/null)
NOW=$(date +%s)
if [ -n "$LAST_MOD" ]; then
    AGE=$((NOW - LAST_MOD))
    if [ $AGE -gt 600 ]; then
        echo "[!!] IVN log unchanged for ${AGE}s (>10min) — possible stall!"
    else
        echo "[OK] IVN log last modified ${AGE}s ago"
    fi
fi

echo ""
echo "=== Monitor done: $(date) ==="
