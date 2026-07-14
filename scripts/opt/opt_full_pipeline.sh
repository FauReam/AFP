#!/bin/bash
# OPT-1.3B Full Replication Pipeline — timeout-protected, skip-on-failure
# Each step: expected_time × 2 timeout → skip to next on failure
set -uo pipefail; cd /home/jiayu/AFP
export HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
V=/home/jiayu/AFP/venv/bin/python3
M=experiments/trained_models_opt; R=experiments/phase0_opt/results
L=experiments/phase0_opt; TS=$(date +%Y%m%d_%H%M%S)
mkdir -p "$M" "$R" "$L"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# Timeout wrapper: $1=label $2=timeout_sec $3+=command
run_with_timeout() {
    local label="$1" timeout_s="$2"; shift 2
    log "  [$label] starting (timeout=${timeout_s}s)"
    if timeout "$timeout_s" "$@" >> "$L/${label}_$TS.log" 2>&1; then
        log "  [$label] OK"
        return 0
    else
        local rc=$?
        if [ $rc -eq 124 ]; then
            log "  [$label] TIMEOUT — skipping"
        else
            log "  [$label] FAILED (exit=$rc) — skipping"
        fi
        return 1
    fi
}

train_one() {
    local domain="$1" seed="$2" lr="$3" epochs="$4"
    local outdir="$M/${domain}_lr${lr}_s${seed}"
    [ -f "$outdir/W_${domain}_final.pt" ] && { log "    [skip] already trained"; return 0; }
    mkdir -p "$outdir"
    run_with_timeout "train_${domain}_s${seed}" 5400 \
        $V -u scripts/opt/train_agent.py --domain "$domain" --lr "$lr" --epochs "$epochs" --output-dir "$outdir"
    # ΔW verify
    if [ -f "$outdir/W_${domain}_final.pt" ]; then
        $V -c "
import torch; from transformers import AutoModel
b=AutoModel.from_pretrained('facebook/opt-1.3b',local_files_only=True,torch_dtype=torch.bfloat16)
bs={k:v.detach().cpu() for k,v in b.state_dict().items()}; del b
m=torch.load('$outdir/W_${domain}_final.pt',map_location='cpu',weights_only=True)
dw=sum((m[k]-bs[k]).float().norm().item()**2 for k in m if k in bs)**0.5
n=sum(bs[k].float().norm().item()**2 for k in bs if k in m)**0.5
print(f'    ΔW={dw/(n+1e-8)*100:.2f}%')
" 2>/dev/null || log "    WARN: ΔW verification failed"
    fi
}

# =====================================================================
log "============================================"
log "OPT-1.3B Full Pipeline"
log "============================================"

# ---- Phase 1: Training (code + medical, std div, 3 seeds each) ----
log ""
log "=== Phase 1: Training code + medical (std div) ~3h ==="
for domain in code medical; do
    for seed in 0 1 2; do
        train_one "$domain" "$seed" "1e-4" 1
    done
done

# ---- Phase 2: Cross-domain LMC (3 seeds) ----
log ""
log "=== Phase 2: Cross-domain LMC ~1h ==="
scan3pt() {
    local a="$1" b="$2" da="$3" db="$4" out="$5" label="$6"
    [ -f "$out" ] && { log "  [skip] $label"; return 0; }
    run_with_timeout "$label" 1200 \
        $V -u scripts/opt/lmc_3pt_scan.py --model-a "$a" --model-b "$b" --domain-a "$da" --domain-b "$db" --output "$out"
}

for seed in 0 1 2; do
    scan3pt "$M/code_lr1e-4_s${seed}" "$M/medical_lr1e-4_s${seed}" code medical \
        "$R/lmc_code_med_opt_s${seed}.json" "cross_s${seed}"
done

# ---- Phase 3: Within-domain LMC (code + medical, 3 pairs each) ----
log ""
log "=== Phase 3: Within-domain LMC ~1.5h ==="
for domain in code medical; do
    for pair in "0 1" "0 2" "1 2"; do
        s1=$(echo $pair|cut -d' ' -f1); s2=$(echo $pair|cut -d' ' -f2)
        scan3pt "$M/${domain}_lr1e-4_s${s1}" "$M/${domain}_lr1e-4_s${s2}" "$domain" "$domain" \
            "$R/lmc_${domain}_within_opt_s${s1}_s${s2}.json" "within_${domain}_s${s1}_s${s2}"
    done
done

# ---- Phase 4: Train math + general (3 seeds each) ----
log ""
log "=== Phase 4: Training math + general ~3h ==="
for domain in math general; do
    for seed in 0 1 2; do
        train_one "$domain" "$seed" "1e-4" 1
    done
done

# ---- Phase 5: Cross-domain math↔general + within-domain ----
log ""
log "=== Phase 5: math/general LMC ~1.5h ==="
for seed in 0 1 2; do
    scan3pt "$M/math_lr1e-4_s${seed}" "$M/general_lr1e-4_s${seed}" math general \
        "$R/lmc_math_gen_opt_s${seed}.json" "cross_mg_s${seed}"
done
for domain in math general; do
    for pair in "0 1" "0 2" "1 2"; do
        s1=$(echo $pair|cut -d' ' -f1); s2=$(echo $pair|cut -d' ' -f2)
        scan3pt "$M/${domain}_lr1e-4_s${s1}" "$M/${domain}_lr1e-4_s${s2}" "$domain" "$domain" \
            "$R/lmc_${domain}_within_opt_s${s1}_s${s2}.json" "within_${domain}_s${s1}_s${s2}"
    done
done

# ---- Phase 6: Per-block divergence analysis ----
log ""
log "=== Phase 6: Per-block divergence ~5min ==="
run_with_timeout "per_block" 600 $V -c "
import torch, json; from transformers import AutoModel
base = AutoModel.from_pretrained('facebook/opt-1.3b',local_files_only=True,torch_dtype=torch.bfloat16)
bs = {k:v.detach().cpu() for k,v in base.state_dict().items()}; del base
results = {}
for domain in ['code','medical']:
    sd = torch.load('experiments/trained_models_opt/${domain}_lr1e-4_s0/W_${domain}_final.pt',map_location='cpu',weights_only=True)
    divs = []
    for blk in range(24):
        bk = [k for k in sd if f'layers.{blk}.' in k]
        dw = sum((sd[k]-bs[k]).float().norm().item()**2 for k in bk if k in bs)**0.5
        n = sum(bs[k].float().norm().item()**2 for k in bk if k in bs)**0.5
        divs.append(round(dw/(n+1e-8)*100,3))
    results[domain] = {'divs': divs, 'mean': sum(divs)/len(divs)}
# Correlation
import numpy as np
r = np.corrcoef(results['code']['divs'], results['medical']['divs'])[0,1]
results['pearson_r'] = round(float(r),4)
with open('experiments/phase0_opt/results/per_block_divergence.json','w') as f: json.dump(results,f,indent=2)
print(f'Per-block done. r={r:.4f}')
"

log ""
log "============================================"
log "OPT-1.3B Pipeline COMPLETE"
log "============================================"
