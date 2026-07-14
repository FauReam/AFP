#!/bin/bash
# ===========================================================================
# Phase 4: Training Trajectory LMC + Gaussian + Layer-Selective Merge (~16 GPU-h)
# Core: continuous ΔW→barrier curve from training checkpoints
# ===========================================================================
set -uo pipefail
cd /home/jiayu/AFP
export HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
VENV=venv/bin/python3; MODELS=experiments/trained_models
RESULTS=experiments/phase0_opt/results; LOGDIR=experiments/phase0_opt
TS=$(date +%Y%m%d_%H%M%S); TRAJ_STEP=40

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

setup_base_vs_checkpoint() {
    # $1=base_pt_path $2=base_head_path $3=ckpt_pt_path $4=ckpt_head_path
    rm -rf "$MODELS/_a" "$MODELS/_b"
    mkdir -p "$MODELS/_a" "$MODELS/_b"
    cp "$1" "$MODELS/_a/W_code_final.pt"
    cp "$2" "$MODELS/_a/W_code_head_final.pt" 2>/dev/null || cp "$MODELS/code_lr1e-4_s0/W_code_head_final.pt" "$MODELS/_a/W_code_head_final.pt"
    cp "$3" "$MODELS/_b/W_medical_final.pt"
    cp "$4" "$MODELS/_b/W_medical_head_final.pt" 2>/dev/null || cp "$MODELS/medical_lr1e-4_s0/W_medical_head_final.pt" "$MODELS/_b/W_medical_head_final.pt"
}

# ===========================================================================
log "============================================"
log "Phase 4: Trajectory LMC + Calibration (~16h)"
log "============================================"

# ---- Step 1: Train code + medical with trajectory checkpoints (~1.5h) ----
log ""
log "=== Step 1: Trajectory training (2 domains, ~1.5h) ==="

for domain in code medical; do
    outdir="$MODELS/${domain}_trajectory"
    if [ -f "$outdir/W_${domain}_final.pt" ]; then
        log "  [skip] $outdir already trained"
    else
        log "  Training $domain with trajectory checkpoints (every ${TRAJ_STEP} steps)..."
        mkdir -p "$outdir"
        $VENV -u scripts/train_agent.py --domain "$domain" --lr 1e-4 \
            --output-dir "$outdir" --trajectory-step "$TRAJ_STEP" --save-every-n-epochs 0 \
            >> "$LOGDIR/train_${domain}_trajectory_$TS.log" 2>&1
        # ΔW verification
        $VENV -c "
import torch; from transformers import AutoModel
base = AutoModel.from_pretrained('facebook/opt-1.3b', local_files_only=True, torch_dtype=torch.bfloat16)
base_sd = {k: v.detach().cpu() for k, v in base.state_dict().items()}; del base
m = torch.load('$outdir/W_${domain}_final.pt', map_location='cpu', weights_only=True)
dw = sum((m[k] - base_sd[k]).float().norm().item()**2 for k in m if k in base_sd)**0.5
n = sum(base_sd[k].float().norm().item()**2 for k in base_sd if k in m)**0.5
print(f'  $outdir ΔW={dw/(n+1e-8)*100:.2f}%')
" || log "  WARN: ΔW verification failed"
        log "  Trajectory training done: $(ls $outdir/trajectory/step_*.pt 2>/dev/null | wc -l) checkpoints saved"
    fi
done

# ---- Step 2: LMC scans for each trajectory checkpoint vs base model (~8h) ----
log ""
log "=== Step 2: Trajectory LMC — ΔW→barrier curve (~8h) ==="

# Load base model once, save to disk for fast reuse
BASE_PT="$MODELS/_base_1.4b.pt"
if [ ! -f "$BASE_PT" ]; then
    log "  Saving base Pythia-1.4B state_dict..."
    $VENV -c "
import torch; from transformers import AutoModel
m = AutoModel.from_pretrained('facebook/opt-1.3b', local_files_only=True, torch_dtype=torch.bfloat16)
torch.save({k: v.detach().cpu() for k, v in m.state_dict().items()}, '$BASE_PT')
print('  Base model saved')
"
fi
HEAD_REF="$MODELS/code_lr1e-4_s0/W_code_head_final.pt"

for domain in code medical; do
    outdir="$MODELS/${domain}_trajectory"
    trajdir="$outdir/trajectory"
    log ""
    log "  --- $domain trajectory LMC ---"
    for ckpt in $(ls "$trajdir"/step_[0-9]*.pt 2>/dev/null | grep -v '_head' | sort -t_ -k2 -n); do
        step=$(basename "$ckpt" .pt | sed 's/step_//')
        out="$RESULTS/lmc_traj_${domain}_step${step}.json"
        [ -f "$out" ] && { log "    [skip] step_$step"; continue; }
        log "    step=$step ($ckpt)"

        # Compute ΔW
        delta=$($VENV -c "
import torch
base = torch.load('$BASE_PT', map_location='cpu', weights_only=True)
ck = torch.load('$ckpt', map_location='cpu', weights_only=True)
dw = sum((ck[k] - base[k]).float().norm().item()**2 for k in ck if k in base)**0.5
n = sum(base[k].float().norm().item()**2 for k in base if k in ck)**0.5
print(f'{dw/(n+1e-8)*100:.4f}')
")
        log "      ΔW=${delta}%"

        # LMC: base vs checkpoint
        head_ckpt="$trajdir/step_${step}_head.pt"
        [ ! -f "$head_ckpt" ] && head_ckpt="$HEAD_REF"
        setup_base_vs_checkpoint "$BASE_PT" "$HEAD_REF" "$ckpt" "$head_ckpt"
        run_lmc "$out" "lmc_traj_${domain}_step${step}"
    done
done

# ---- Step 3: Fine-grained Gaussian perturbation (~2.5h) ----
log ""
log "=== Step 3: Gaussian perturbation at 5 ΔW levels (~2.5h) ==="

for dw_target in 0.5 1.0 2.0 4.0 8.0; do
    out="$RESULTS/noise_gaussian_dw${dw_target}.json"
    [ -f "$out" ] && { log "  [skip] ΔW=${dw_target}%"; continue; }
    log "  Gaussian ΔW≈${dw_target}%"

    $VENV -c "
import torch, os
from transformers import AutoModel
base = AutoModel.from_pretrained('facebook/opt-1.3b', local_files_only=True, torch_dtype=torch.bfloat16)
base_sd = {k: v.detach().cpu() for k, v in base.state_dict().items()}
base_norm = sum(v.float().norm().item()**2 for v in base_sd.values())**0.5
del base; torch.cuda.empty_cache()

# Per-param Gaussian noise, scaled to target ΔW
torch.manual_seed(42)
noise_sd = {}
for k, v in base_sd.items():
    std = v.float().std()
    noise_sd[k] = v.float() + torch.randn_like(v.float()) * std

curr_dw = sum((noise_sd[k] - base_sd[k]).float().norm().item()**2 for k in base_sd)**0.5 / base_norm * 100
scale = $dw_target / curr_dw
for k in noise_sd:
    noise_sd[k] = (base_sd[k].float() + (noise_sd[k] - base_sd[k].float()) * scale).to(base_sd[k].dtype)
final_dw = sum((noise_sd[k] - base_sd[k]).float().norm().item()**2 for k in base_sd)**0.5 / base_norm * 100
print(f'  Gaussian ΔW={final_dw:.2f}% (target=$dw_target)')

os.makedirs('$MODELS/_a', exist_ok=True); os.makedirs('$MODELS/_b', exist_ok=True)
torch.save(base_sd, '$MODELS/_a/W_code_final.pt')
torch.save(noise_sd, '$MODELS/_b/W_medical_final.pt')
import shutil
shutil.copy('$MODELS/code_lr1e-4_s0/W_code_head_final.pt', '$MODELS/_a/W_code_head_final.pt')
shutil.copy('$MODELS/medical_lr1e-4_s0/W_medical_head_final.pt', '$MODELS/_b/W_medical_head_final.pt')
" >> "$LOGDIR/noise_gaussian_dw${dw_target}_$TS.log" 2>&1

    run_lmc "$out" "noise_gaussian_dw${dw_target}"
done

# ---- Step 4: Layer-selective merge (~2h) ----
log ""
log "=== Step 4: Layer-selective interpolation — which layers drive barrier? (~2h) ==="

for layers in "early:0-7" "mid:8-15" "late:16-23"; do
    label=$(echo $layers | cut -d: -f1)
    lrange=$(echo $layers | cut -d: -f2)
    lstart=$(echo $lrange | cut -d- -f1); lend=$(echo $lrange | cut -d- -f2)
    out="$RESULTS/lmc_layers_${label}.json"
    [ -f "$out" ] && { log "  [skip] layers=$label"; continue; }
    log "  Layer-selective: $label ($lrange)"

    $VENV -c "
import torch, os
sd_code = torch.load('$MODELS/code_lr1e-4_s0/W_code_final.pt', map_location='cpu', weights_only=True)
sd_med  = torch.load('$MODELS/medical_lr1e-4_s0/W_medical_final.pt', map_location='cpu', weights_only=True)

# Create merged model: interpolate only layers $lstart-$lend, keep code model's other layers
sd_merged = {}
for k in sd_code:
    layer_match = False
    for l in range($lstart, $lend + 1):
        if f'layers.{l}.' in k:
            layer_match = True; break
    if layer_match:
        sd_merged[k] = 0.5 * sd_code[k].float() + 0.5 * sd_med[k].float()
    else:
        sd_merged[k] = sd_code[k]

os.makedirs('$MODELS/_a', exist_ok=True); os.makedirs('$MODELS/_b', exist_ok=True)
torch.save(sd_code, '$MODELS/_a/W_code_final.pt')
torch.save(sd_merged, '$MODELS/_b/W_medical_final.pt')
import shutil
shutil.copy('$MODELS/code_lr1e-4_s0/W_code_head_final.pt', '$MODELS/_a/W_code_head_final.pt')
shutil.copy('$MODELS/code_lr1e-4_s0/W_code_head_final.pt', '$MODELS/_b/W_medical_head_final.pt')
print(f'  Layers {label} merged, saved')
" >> "$LOGDIR/lmc_layers_${label}_$TS.log" 2>&1

    run_lmc "$out" "lmc_layers_${label}"
done

# Also: all-layers merge (standard cross-domain, already done as lmc_lr1e-4_s0)
# Also: no-layers merge (identical copy, already done as noise_identical)

# ---- Step 5: Fixed E8 Gaussian (1.5%, 8%) using real head files (~1h) ----
log ""
log "=== Step 5: Fixed E8 Gaussian (with real head files) (~1h) ==="

for dw_target in 1.5 8.0; do
    out="$RESULTS/noise_gaussian_fixed_dw${dw_target}.json"
    [ -f "$out" ] && { log "  [skip] Fixed ΔW=${dw_target}%"; continue; }
    log "  Fixed Gaussian ΔW≈${dw_target}%"

    $VENV -c "
import torch, os
from transformers import AutoModel
base = AutoModel.from_pretrained('facebook/opt-1.3b', local_files_only=True, torch_dtype=torch.bfloat16)
base_sd = {k: v.detach().cpu() for k, v in base.state_dict().items()}
base_norm = sum(v.float().norm().item()**2 for v in base_sd.values())**0.5
del base; torch.cuda.empty_cache()

torch.manual_seed(42)
noise_sd = {}
for k, v in base_sd.items():
    noise_sd[k] = v.float() + torch.randn_like(v.float()) * v.float().std()
curr_dw = sum((noise_sd[k] - base_sd[k]).float().norm().item()**2 for k in base_sd)**0.5 / base_norm * 100
scale = $dw_target / curr_dw
for k in noise_sd:
    noise_sd[k] = (base_sd[k].float() + (noise_sd[k] - base_sd[k].float()) * scale).to(base_sd[k].dtype)
final_dw = sum((noise_sd[k] - base_sd[k]).float().norm().item()**2 for k in base_sd)**0.5 / base_norm * 100
print(f'  Fixed Gaussian ΔW={final_dw:.2f}%')

os.makedirs('$MODELS/_a', exist_ok=True); os.makedirs('$MODELS/_b', exist_ok=True)
torch.save(base_sd, '$MODELS/_a/W_code_final.pt')
torch.save(noise_sd, '$MODELS/_b/W_medical_final.pt')
import shutil
shutil.copy('$MODELS/code_lr1e-4_s0/W_code_head_final.pt', '$MODELS/_a/W_code_head_final.pt')
shutil.copy('$MODELS/medical_lr1e-4_s0/W_medical_head_final.pt', '$MODELS/_b/W_medical_head_final.pt')
" >> "$LOGDIR/noise_gaussian_fixed_dw${dw_target}_$TS.log" 2>&1

    run_lmc "$out" "noise_gaussian_fixed_dw${dw_target}"
done

# ---- Cleanup ----
rm -rf "$MODELS/_a" "$MODELS/_b" "$MODELS/code_e1" "$MODELS/medical_e1" "$RESULTS/lmc_barrier_c1m1.json"

log ""
log "============================================"
log "Phase 4 COMPLETE"
log "============================================"
