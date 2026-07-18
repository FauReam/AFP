#!/bin/bash
# GPT-Neo-1.3B pipeline v5 — clean, no data swap needed, expandable_segments
set -uo pipefail; cd /home/jiayu/AFP
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
V=/home/jiayu/AFP/venv/bin/python3; TS=$(date +%Y%m%d_%H%M%S)
M=/home/jiayu/AFP/experiments/trained_models_gptneo
R=/home/jiayu/AFP/experiments/phase0_gptneo/results
L=/home/jiayu/AFP/experiments/phase0_gptneo; MID=EleutherAI/gpt-neo-1.3B
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "=== GPT-Neo v5: Training (6 models, 2-epoch) ==="
for domain in code medical; do for seed in 0 1 2; do
    outdir="$M/${domain}_lr1e-4_s${seed}"
    if [ -f "$outdir/W_${domain}_final.pt" ]; then log "  [skip] $domain s$seed"; continue; fi
    mkdir -p "$outdir"
    log "  Training $domain s$seed..."
    $V -u scripts/train_agent.py --domain "$domain" --lr 1e-4 --epochs 2 --output-dir "$outdir" --model-id "$MID" >> "$L/train_${domain}_s${seed}_$TS.log" 2>&1
    $V -c "import torch; torch.cuda.empty_cache()" 2>/dev/null
done; done
log "Training done"

log "=== Cross-domain LMC ==="
for s in 0 1 2; do
    out="$R/lmc_gptneo_cross_s${s}.json"; [ -f "$out" ] && continue
    $V -u scripts/lmc_3pt_scan.py --model-a "$M/code_lr1e-4_s${s}" --model-b "$M/medical_lr1e-4_s${s}" --domain-a code --domain-b medical --output "$out" >> "$L/lmc_cross_s${s}_$TS.log" 2>&1
done

log "=== Within-domain LMC ==="
for domain in code medical; do for p in "0 1" "0 2" "1 2"; do
    s1=$(echo $p|cut -d" " -f1); s2=$(echo $p|cut -d" " -f2)
    out="$R/lmc_gptneo_${domain}_within_s${s1}_s${s2}.json"; [ -f "$out" ] && continue
    $V -u scripts/lmc_3pt_scan.py --model-a "$M/${domain}_lr1e-4_s${s1}" --model-b "$M/${domain}_lr1e-4_s${s2}" --domain-a "$domain" --domain-b "$domain" --output "$out" >> "$L/lmc_${domain}_s${s1}_s${s2}_$TS.log" 2>&1
done; done

log "=== GPT-Neo v5 DONE ==="
