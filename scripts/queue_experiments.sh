#!/bin/bash
# ===========================================================================
# Queue multiple IVN experiments across the divergence spectrum.
#
# Runs AFTER train_and_run_phase0.sh completes. Uses trained checkpoints
# to test AFP/IVN at different model-divergence levels.
#
# Usage:
#   nohup bash scripts/queue_experiments.sh > experiments/queue.log 2>&1 &
# ===========================================================================
set -euo pipefail

cd "$(dirname "$0")/.."
export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
VENV=venv/bin/python3

RESULTS_DIR="experiments/phase0_ivn"
WEIGHTS_DIR="experiments/trained_models"
mkdir -p "$RESULTS_DIR/results"

echo "=============================================="
echo " Experiment Queue — $(date)"
echo "=============================================="
echo ""

# ---------------------------------------------------------------------------
# Helper: run one IVN experiment
# ---------------------------------------------------------------------------
run_ivn() {
    local label=$1         # e.g. "c1m1"
    local code_ckpt=$2     # e.g. "code_e1"
    local med_ckpt=$3      # e.g. "medical_e1"
    local extra_args=${4:-}

    echo "--- [$label] IVN: $code_ckpt + $med_ckpt ---"

    # Verify checkpoints exist
    if [ ! -f "$WEIGHTS_DIR/$code_ckpt/W_code_final.pt" ]; then
        echo "  [SKIP] $code_ckpt not found"
        return 0
    fi
    if [ ! -f "$WEIGHTS_DIR/$med_ckpt/W_medical_final.pt" ]; then
        echo "  [SKIP] $med_ckpt not found"
        return 0
    fi

    # Create symlinks
    rm -f "$WEIGHTS_DIR/code" "$WEIGHTS_DIR/medical"
    ln -sf "$code_ckpt" "$WEIGHTS_DIR/code"
    ln -sf "$med_ckpt" "$WEIGHTS_DIR/medical"

    RUN_LOG="$RESULTS_DIR/logs/queue_${label}_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p "$(dirname "$RUN_LOG")"

    $VENV -u scripts/run_ivn_phase0.py \
        --teacher EleutherAI/pythia-1.4b \
        --student EleutherAI/pythia-1.4b \
        --base-model EleutherAI/pythia-1.4b \
        --teacher-domain code \
        --student-domain medical \
        --weights "$WEIGHTS_DIR" \
        --importance magnitude_l2 \
        --gate rational \
        --tau 0.5 \
        --max-rounds 30 \
        $extra_args \
        >> "$RUN_LOG" 2>&1

    local rc=$?
    if [ $rc -eq 0 ]; then
        # Copy result with label
        if [ -f "$RESULTS_DIR/ivn_results.json" ]; then
            cp "$RESULTS_DIR/ivn_results.json" "$RESULTS_DIR/results/${label}_results.json"
            echo "  [OK] saved -> results/${label}_results.json"
            # Quick summary
            $VENV -c "
import json
r = json.load(open('$RESULTS_DIR/results/${label}_results.json'))
cos = r.get('importance_cosine', float('nan'))
fed = r.get('fedavg',{}).get('net', 0)
afp = r.get('afp_oneshot',{}).get('net', 0)
ivn = r.get('ivn',{}).get('net', 0)
print(f'  cos={cos:.4f}  FedAvg={fed:+.4f}  AFP={afp:+.4f}  IVN={ivn:+.4f}')
" || true
        fi
    else
        echo "  [FAIL] exit code $rc — see $RUN_LOG"
    fi
    echo ""
}

# ---------------------------------------------------------------------------
# Experiment queue
# ---------------------------------------------------------------------------

echo "=== Divergence spectrum: code vs medical at different training levels ==="
echo ""

# E1: Lowest divergence (both 1 epoch)
run_ivn "c1m1" "code_e1" "medical_e1"

# E2: Medium divergence (code 1 epoch, medical 3 epochs)
run_ivn "c1m3" "code_e1" "medical_e3"

# E3: Higher divergence (code 1 epoch, medical 5 epochs)
run_ivn "c1m5" "code_e1" "medical_e5"

# E4-E6: If current pipeline produced code_e2+, include those
for e in 2 3 4 5; do
    run_ivn "c${e}m${e}" "code_e${e}" "medical_e${e}"
done

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
echo "=============================================="
echo " Results Summary — $(date)"
echo "=============================================="
printf "%-10s %8s %10s %10s %10s\n" "Experiment" "Cosine" "FedAvg" "AFP" "IVN"
echo "--------------------------------------------------------"

for f in "$RESULTS_DIR"/results/*_results.json; do
    [ -f "$f" ] || continue
    label=$(basename "$f" _results.json)
    $VENV -c "
import json
r = json.load(open('$f'))
cos = r.get('importance_cosine', float('nan'))
fed = r.get('fedavg',{}).get('net', 0)
afp = r.get('afp_oneshot',{}).get('net', 0)
ivn = r.get('ivn',{}).get('net', 0)
print(f'{$label:<10} {cos:8.4f} {fed:+10.4f} {afp:+10.4f} {ivn:+10.4f}')
" 2>/dev/null || true
done

echo ""
echo "Queue complete — $(date)"
