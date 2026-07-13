#!/bin/bash
set -uo pipefail
cd /home/jiayu/AFP
export HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
VENV=venv/bin/python3; MODELS=experiments/trained_models
RESULTS=experiments/phase0_ivn/results; LOGDIR=experiments/phase0_training
TS=$(date +%Y%m%d_%H%M%S)

log() { echo "[$(date +%H:%M:%S)] $*"; }

run_lmc() {
    local out="$1" label="$2"
    rm -f "$RESULTS/lmc_barrier_c1m1.json" "$MODELS/code_e1" "$MODELS/medical_e1"
    (cd "$MODELS" && ln -sfn _a code_e1 && ln -sfn _b medical_e1)
    $VENV -u scripts/lmc_barrier_scan.py >> "$LOGDIR/${label}_$TS.log" 2>&1
    if [ -f "$RESULTS/lmc_barrier_c1m1.json" ]; then
        cp "$RESULTS/lmc_barrier_c1m1.json" "$out"
        $VENV -c "
import json; d=json.load(open('$out')); r=d['results']
c0=r[0]['loss_code']; cm=max(x['loss_code'] for x in r); c1=r[-1]['loss_code']
m0=r[0]['loss_med']; mm=max(x['loss_med'] for x in r); m1=r[-1]['loss_med']
print(f'  $label: bar_code={cm-(c0+c1)/2:.4f}  bar_med={mm-(m0+m1)/2:.4f}')
" || log "  WARN: parse failed"
    else
        log "  ERROR: LMC scan failed for $label"
    fi
}

setup_pair() {
    # $1=dir1 $2=dir2 $3=domain1 $4=domain2
    rm -rf "$MODELS/_a" "$MODELS/_b"
    mkdir -p "$MODELS/_a" "$MODELS/_b"
    cp "$1/W_$3_final.pt"      "$MODELS/_a/W_code_final.pt"
    cp "$1/W_$3_head_final.pt" "$MODELS/_a/W_code_head_final.pt" 2>/dev/null || true
    cp "$2/W_$4_final.pt"      "$MODELS/_b/W_medical_final.pt"
    cp "$2/W_$4_head_final.pt" "$MODELS/_b/W_medical_head_final.pt" 2>/dev/null || true
}

log "=== Phase 3: math/general LMC scans (~2.5h) ==="

# Cross-domain: math ↔ general (3 seeds)
for seed in 0 1 2; do
    out="$RESULTS/lmc_math_general_s${seed}.json"
    [ -f "$out" ] && { log "  [skip] $out"; continue; }
    log "  Cross: math_s$seed ↔ general_s$seed"
    setup_pair "$MODELS/math_lr1e-4_s${seed}" "$MODELS/general_lr1e-4_s${seed}" "math" "general"
    run_lmc "$out" "lmc_math_general_s${seed}"
done

# Math within-domain (3 pairs)
for pair in "0 1" "0 2" "1 2"; do
    s1=$(echo $pair | cut -d' ' -f1); s2=$(echo $pair | cut -d' ' -f2)
    out="$RESULTS/lmc_math_s${s1}_s${s2}.json"
    [ -f "$out" ] && { log "  [skip] $out"; continue; }
    log "  Math within: s$s1 ↔ s$s2"
    setup_pair "$MODELS/math_lr1e-4_s${s1}" "$MODELS/math_lr1e-4_s${s2}" "math" "math"
    run_lmc "$out" "lmc_math_s${s1}_s${s2}"
done

# General within-domain (3 pairs)
for pair in "0 1" "0 2" "1 2"; do
    s1=$(echo $pair | cut -d' ' -f1); s2=$(echo $pair | cut -d' ' -f2)
    out="$RESULTS/lmc_general_s${s1}_s${s2}.json"
    [ -f "$out" ] && { log "  [skip] $out"; continue; }
    log "  General within: s$s1 ↔ s$s2"
    setup_pair "$MODELS/general_lr1e-4_s${s1}" "$MODELS/general_lr1e-4_s${s2}" "general" "general"
    run_lmc "$out" "lmc_general_s${s1}_s${s2}"
done

rm -rf "$MODELS/_a" "$MODELS/_b" "$MODELS/code_e1" "$MODELS/medical_e1" "$RESULTS/lmc_barrier_c1m1.json"
log "=== Phase 3 COMPLETE ==="
