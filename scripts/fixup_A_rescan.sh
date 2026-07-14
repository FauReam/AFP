#!/bin/bash
# Fixup: re-run Experiment A LMC scans with corrected paths
set -uo pipefail; cd /home/jiayu/AFP
export HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
V=venv/bin/python3; M=experiments/trained_models; R=experiments/phase0_ivn/results
L=experiments/phase0_training; TS=$(date +%Y%m%d_%H%M%S)
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "=== A fixup: medical 2-epoch within-domain scans ==="
for p in "10 11" "10 12" "11 12"; do
    s1=$(echo $p|cut -d' ' -f1); s2=$(echo $p|cut -d' ' -f2)
    out="$R/lmc_medical_2ep_s${s1}_s${s2}.json"
    [ -f "$out" ] && { log "  [skip] s${s1}_s${s2}"; continue; }
    log "  s$s1 ↔ s$s2"
    $V -u scripts/lmc_3pt_scan.py \
        --model-a "$M/medical_lr1e-4_2ep_s${s1}" --model-b "$M/medical_lr1e-4_2ep_s${s2}" \
        --domain-a medical --domain-b medical \
        --output "$out" >> "$L/lmc_med_2ep_s${s1}_s${s2}_$TS.log" 2>&1
    if [ -f "$out" ]; then
        $V -c "import json;d=json.load(open('$out'));r=d['results'];m0=r[0]['loss_med'];m1=r[-1]['loss_med'];mm=max(x['loss_med'] for x in r);print(f'  => bar_med={mm-(m0+m1)/2:.4f}')"
    else
        log "  ERROR"
    fi
done
log "DONE"
