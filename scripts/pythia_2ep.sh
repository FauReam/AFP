#!/bin/bash
set -uo pipefail; cd /home/jiayu/AFP
export HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
V=/home/jiayu/AFP/venv/bin/python3; TS=$(date +%Y%m%d_%H%M%S)
M=/home/jiayu/AFP/experiments/trained_models
R=/home/jiayu/AFP/experiments/phase0_ivn/results
L=/home/jiayu/AFP/experiments/phase0_training
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "=== Pythia 2-epoch training (6 models) ==="
for domain in code medical; do
  for seed in 0 1 2; do
    outdir="$M/${domain}_2ep_lr1e-4_s${seed}"
    [ -f "$outdir/W_${domain}_final.pt" ] && { log "  [skip] $domain s$seed"; continue; }
    mkdir -p "$outdir"
    log "  Training $domain s$seed..."
    $V -u scripts/train_agent.py --domain "$domain" --lr 1e-4 --epochs 2 --output-dir "$outdir" >> "$L/py2ep_${domain}_s${seed}_$TS.log" 2>&1
  done
done

log "=== Cross-domain LMC ==="
scan3() {
    $V -u scripts/lmc_3pt_scan.py --model-a "$1" --model-b "$2" --domain-a "$3" --domain-b "$4" --output "$5" >> "$6" 2>&1
}
for s in 0 1 2; do
    out="$R/lmc_py2ep_cross_s${s}.json"
    [ -f "$out" ] && { log "  [skip] cross_s$s"; continue; }
    log "  cross_s$s"
    scan3 "$M/code_2ep_lr1e-4_s${s}" "$M/medical_2ep_lr1e-4_s${s}" code medical "$out" "$L/py2ep_lmc_cross_s${s}_$TS.log"
done

log "=== Within-domain LMC ==="
for domain in code medical; do
    for p in "0 1" "0 2" "1 2"; do
        s1=$(echo $p|cut -d" " -f1); s2=$(echo $p|cut -d" " -f2)
        out="$R/lmc_py2ep_${domain}_within_s${s1}_s${s2}.json"
        [ -f "$out" ] && { log "  [skip] ${domain}_s${s1}_s${s2}"; continue; }
        log "  ${domain}_s${s1}_s${s2}"
        scan3 "$M/${domain}_2ep_lr1e-4_s${s1}" "$M/${domain}_2ep_lr1e-4_s${s2}" "$domain" "$domain" "$out" "$L/py2ep_lmc_${domain}_s${s1}_s${s2}_$TS.log"
    done
done

log "=== DONE ==="
