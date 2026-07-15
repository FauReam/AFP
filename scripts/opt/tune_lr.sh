#!/bin/bash
# OPT LR sweep: find divergence level matching Pythia (~1.4% ΔW, self-loss<0.7)
set -uo pipefail; cd /home/jiayu/AFP
export HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
V=/home/jiayu/AFP/venv/bin/python3
M=experiments/trained_models_opt; L=experiments/phase0_opt; TS=$(date +%Y%m%d_%H%M%S)
mkdir -p "$M" "$L"

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "=== OPT LR Sweep: code domain ==="
log "Target: ΔW≈1.4%, self-loss<0.7 (matching Pythia)"

for lr in 5e-5 1e-4 3e-4; do
    for seed in 10; do  # single seed for sweep
        outdir="$M/tune_code_lr${lr}_s${seed}"
        [ -f "$outdir/W_code_final.pt" ] && { log "  [skip] lr=$lr"; continue; }
        mkdir -p "$outdir"
        log "  Training lr=$lr seed=$seed..."
        $V -u scripts/opt/train_agent.py --domain code --lr "$lr" --output-dir "$outdir" --epochs 1 \
            >> "$L/tune_lr${lr}_$TS.log" 2>&1
        if [ -f "$outdir/W_code_final.pt" ]; then
            $V -c "
import torch; from transformers import AutoModel
base=AutoModel.from_pretrained('facebook/opt-1.3b',local_files_only=True,torch_dtype=torch.bfloat16)
bs={k:v.detach().cpu() for k,v in base.state_dict().items()}; bn=sum(v.float().norm().item()**2 for v in bs.values())**0.5
del base
m=torch.load('$outdir/W_code_final.pt',map_location='cpu',weights_only=True)
dw=sum((m[k]-bs[k]).float().norm().item()**2 for k in m if k in bs)**0.5
print(f'  => ΔW={dw/(bn+1e-8)*100:.2f}%')
" 2>/dev/null
        fi
    done
done

log "LR sweep done. Check self-loss via LMC scan of best model."
