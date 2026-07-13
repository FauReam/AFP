#!/bin/bash
# ===========================================================================
# Phase 1 Batch: ICLR Sprint MUST + SHOULD experiments (~14-16 GPU-hours)
# Usage: nohup bash scripts/phase1_batch.sh >> experiments/phase0_training/phase1_batch_$(date +%Y%m%d_%H%M%S).log 2>&1 &
# ===========================================================================
set -uo pipefail
# NOTE: NO 'set -e' — individual experiment failure must NOT kill the batch.
# Each step has its own error handling; failed steps are logged and skipped on rerun.
cd /home/jiayu/AFP
export HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
VENV=venv/bin/python3; MODELS=experiments/trained_models
RESULTS=experiments/phase0_ivn/results; LOGDIR=experiments/phase0_training
TS=$(date +%Y%m%d_%H%M%S)

log() { echo "[$(date +%H:%M:%S)] $*"; }

clean_symlinks() {
    # Only remove symlinks and stale result — NOT _a/_b/_p1/_p2 dirs
    # (those are managed by callers for within-domain / noise floor scans)
    rm -f "$MODELS/code_e1" "$MODELS/medical_e1" "$RESULTS/lmc_barrier_c1m1.json"
}

run_lmc_scan() {
    # $1=code_dir_name  $2=med_dir_name  $3=output_json_path  $4=log_label
    local cdir="$1" mdir="$2" out="$3" label="$4"
    clean_symlinks
    (cd "$MODELS" && ln -sfn "$cdir" code_e1 && ln -sfn "$mdir" medical_e1)
    rm -f "$RESULTS/lmc_barrier_c1m1.json"
    $VENV -u scripts/lmc_barrier_scan.py >> "$LOGDIR/${label}_$TS.log" 2>&1
    if [ -f "$RESULTS/lmc_barrier_c1m1.json" ]; then
        cp "$RESULTS/lmc_barrier_c1m1.json" "$out"
        $VENV -c "
import json; d=json.load(open('$out')); r=d['results']
c0=r[0]['loss_code']; cm=max(x['loss_code'] for x in r); c1=r[-1]['loss_code']
m0=r[0]['loss_med']; mm=max(x['loss_med'] for x in r); m1=r[-1]['loss_med']
print(f'  {label}: bar_code={cm-(c0+c1)/2:.4f}  bar_med={mm-(m0+m1)/2:.4f}')
" || log "  WARN: parse failed for $out"
    else
        log "  ERROR: LMC scan failed for $label"
    fi
}

train_model() {
    # $1=domain $2=lr $3=seed_suffix
    local domain="$1" lr="$2" suffix="$3"
    local outdir="$MODELS/${domain}_lr${lr}_s${suffix}"
    if [ -f "$outdir/W_${domain}_final.pt" ]; then
        log "  [skip] $outdir already exists"
        return 0
    fi
    log "  Training: $outdir"
    mkdir -p "$outdir"
    $VENV -u scripts/train_agent.py --domain "$domain" --lr "$lr" --output-dir "$outdir" \
        >> "$LOGDIR/train_${domain}_lr${lr}_s${suffix}_$TS.log" 2>&1
    # Verify ΔW > 0.1%
    $VENV -c "
import torch
from transformers import AutoModel
base = AutoModel.from_pretrained('EleutherAI/pythia-1.4b', local_files_only=True, torch_dtype=torch.bfloat16)
base_sd = {k: v.detach().cpu() for k, v in base.state_dict().items()}
del base
m = torch.load('$outdir/W_${domain}_final.pt', map_location='cpu', weights_only=True)
dw = sum((m[k] - base_sd[k]).float().norm().item()**2 for k in m if k in base_sd)**0.5
n = sum(base_sd[k].float().norm().item()**2 for k in base_sd if k in m)**0.5
pct = dw/(n+1e-8)*100
if pct < 0.1:
    print(f'BUG: $outdir ΔW={pct:.3f}% < 0.1% — model may be untrained!')
    import sys; sys.exit(1)
print(f'  $outdir ΔW={pct:.2f}% OK')
" || { log "  FATAL: ΔW verification failed for $outdir"; return 1; }
}

# ===========================================================================
echo "============================================"
log "Phase 1 Batch Start"
echo "============================================"

# ---- Step 0: Pre-tokenize math + general data (needed for training) ----
log ""
log "=== Step 0: Pre-tokenize math + general data ==="
for domain in math general; do
    cache="data/versaprm/train_${domain}_pythia_L256.pt"
    if [ -f "$cache" ]; then
        log "  [skip] $cache already cached"
    else
        log "  Tokenizing $domain..."
        $VENV -c "
from scripts.train_agent import prepare_data
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('EleutherAI/pythia-1.4b', local_files_only=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
prepare_data('$domain', tok)
print('  $domain tokenized')
" >> "$LOGDIR/tokenize_${domain}_$TS.log" 2>&1
        log "  $domain tokenization done"
    fi
done

# ===========================================================================
# Phase 1a: LMC scans of EXISTING models (no training needed)
# ===========================================================================
log ""
log "=== Phase 1a: LMC scans of existing models (~3 GPU-hours) ==="

# ---- E1: lr=2e-4 and 3e-4 cross-domain LMC (6 scans) ----
log ""
log "--- E1: lr=2e-4, 3e-4 cross-domain LMC ---"
for lr in 2e-4 3e-4; do
    for seed in 0 1 2; do
        code_dir="code_lr${lr}_s${seed}"
        med_dir="medical_lr${lr}_s${seed}"
        out="$RESULTS/lmc_lr${lr}_s${seed}.json"
        if [ -f "$out" ]; then
            log "  [skip] $out exists"
            continue
        fi
        log "  Scanning: $code_dir ↔ $med_dir"
        run_lmc_scan "$code_dir" "$med_dir" "$out" "lmc_lr${lr}_s${seed}"
    done
done

# ---- E10: High-div within-domain baselines (6 scans) ----
log ""
log "--- E10: High-div within-domain baselines ---"
for domain in code medical; do
    for pair in "0 1" "0 2" "1 2"; do
        s1=$(echo $pair | cut -d' ' -f1); s2=$(echo $pair | cut -d' ' -f2)
        dir1="${MODELS}/${domain}_lr5e-4_s${s1}"
        dir2="${MODELS}/${domain}_lr5e-4_s${s2}"
        out="$RESULTS/lmc_${domain}_high_s${s1}_s${s2}.json"
        if [ -f "$out" ]; then
            log "  [skip] $out exists"
            continue
        fi
        log "  Within-domain ${domain} high-div s$s1 ↔ s$s2"

        # Setup: copy both to temp dirs with code/medical naming for LMC script
        rm -rf "$MODELS/_a" "$MODELS/_b"
        mkdir -p "$MODELS/_a" "$MODELS/_b"
        cp "$dir1/W_${domain}_final.pt" "$MODELS/_a/W_code_final.pt"
        cp "$dir1/W_${domain}_head_final.pt" "$MODELS/_a/W_code_head_final.pt" 2>/dev/null || true
        cp "$dir2/W_${domain}_final.pt" "$MODELS/_b/W_medical_final.pt"
        cp "$dir2/W_${domain}_head_final.pt" "$MODELS/_b/W_medical_head_final.pt" 2>/dev/null || true

        run_lmc_scan "_a" "_b" "$out" "lmc_${domain}_high_s${s1}_s${s2}"
    done
done

# ---- E8: Gaussian perturbation baseline (~1 GPU-hour) ----
log ""
log "--- E8: Gaussian perturbation baseline ---"
for dw_target in 1.5 8.0; do
    out="$RESULTS/noise_gaussian_dw${dw_target}.json"
    if [ -f "$out" ]; then
        log "  [skip] $out exists"
        continue
    fi
    log "  Gaussian noise ΔW≈${dw_target}%"

    $VENV -c "
import torch, os, json
from transformers import AutoModel

base = AutoModel.from_pretrained('EleutherAI/pythia-1.4b', local_files_only=True, torch_dtype=torch.bfloat16)
base_sd = {k: v.detach().cpu() for k, v in base.state_dict().items()}
base_norm = sum(v.float().norm().item()**2 for v in base_sd.values())**0.5
del base; torch.cuda.empty_cache()

# Create perturbed model by adding Gaussian noise scaled to target ΔW
torch.manual_seed(42)
noise_sd = {}
for k, v in base_sd.items():
    noise = torch.randn_like(v.float()) * v.float().std()
    noise_sd[k] = (v.float() + noise).to(v.dtype)

# Scale noise to match target ΔW
curr_dw = sum((noise_sd[k] - base_sd[k]).float().norm().item()**2 for k in base_sd)**0.5 / base_norm * 100
scale = $dw_target / curr_dw
for k in noise_sd:
    noise_sd[k] = (base_sd[k].float() + (noise_sd[k] - base_sd[k].float()) * scale).to(base_sd[k].dtype)

# Verify
final_dw = sum((noise_sd[k] - base_sd[k]).float().norm().item()**2 for k in base_sd)**0.5 / base_norm * 100
print(f'  Gaussian ΔW={final_dw:.2f}% (target=${dw_target}%)')

os.makedirs('$MODELS/_p1', exist_ok=True)
os.makedirs('$MODELS/_p2', exist_ok=True)
torch.save(base_sd, '$MODELS/_p1/W_code_final.pt')
torch.save(noise_sd, '$MODELS/_p2/W_medical_final.pt')
import shutil; shutil.copy('$MODELS/code_lr1e-4_s0/W_code_head_final.pt', '$MODELS/_p1/W_code_head_final.pt')
shutil.copy('$MODELS/medical_lr1e-4_s0/W_medical_head_final.pt', '$MODELS/_p2/W_medical_head_final.pt')
print('  Gaussian models saved')
" >> "$LOGDIR/noise_gaussian_dw${dw_target}_$TS.log" 2>&1

    run_lmc_scan "_p1" "_p2" "$out" "noise_gaussian_dw${dw_target}"
done

# ===========================================================================
# Phase 1b: Random init recalibration (E2, ~1 GPU-hour)
# ===========================================================================
log ""
log "=== Phase 1b: Random init recalibration with 3 seeds (~1 GPU-hour) ==="

for s in 0 1 2; do
    out="$RESULTS/noise_random_s${s}.json"
    if [ -f "$out" ]; then
        log "  [skip] $out exists"
        continue
    fi
    log "  Random init seed=$s"

    $VENV -c "
import torch, os, json
from transformers import AutoModel
import torch.nn as nn

pretrained = AutoModel.from_pretrained('EleutherAI/pythia-1.4b', local_files_only=True, torch_dtype=torch.bfloat16)
pretrained_sd = {k: v.detach().cpu() for k, v in pretrained.state_dict().items()}
del pretrained; torch.cuda.empty_cache()

torch.manual_seed($s)
random_model = AutoModel.from_pretrained('EleutherAI/pythia-1.4b', local_files_only=True, torch_dtype=torch.bfloat16)
for p in random_model.parameters():
    if len(p.shape) >= 2: nn.init.xavier_uniform_(p)
    else: nn.init.uniform_(p, -0.1, 0.1)
random_sd = {k: v.detach().cpu() for k, v in random_model.state_dict().items()}
del random_model; torch.cuda.empty_cache()

os.makedirs('$MODELS/_p1', exist_ok=True)
os.makedirs('$MODELS/_p2', exist_ok=True)
torch.save(pretrained_sd, '$MODELS/_p1/W_code_final.pt')
torch.save(random_sd,      '$MODELS/_p2/W_medical_final.pt')
import shutil
shutil.copy('$MODELS/code_lr1e-4_s0/W_code_head_final.pt', '$MODELS/_p1/W_code_head_final.pt')
shutil.copy('$MODELS/medical_lr1e-4_s0/W_medical_head_final.pt', '$MODELS/_p2/W_medical_head_final.pt')
print(f'  Random models saved (seed=$s)')
" >> "$LOGDIR/noise_random_s${s}_$TS.log" 2>&1

    run_lmc_scan "_p1" "_p2" "$out" "noise_random_s${s}"
done

# ===========================================================================
# Phase 2: Train new models (~8 GPU-hours)
# ===========================================================================
log ""
log "=== Phase 2: Training new models (~8 GPU-hours) ==="

# ---- E6: Train math + general domains (standard divergence, 3 seeds each) ----
log ""
log "--- E6: math + general domains (standard divergence) ---"
for domain in math general; do
    for seed in 0 1 2; do
        train_model "$domain" "1e-4" "$seed"
    done
done

# ---- E3: Additional high-div medical seeds ----
log ""
log "--- E3: Additional high-div medical seeds 3, 4 ---"
for seed in 3 4; do
    train_model "medical" "5e-4" "$seed"
done

# ---- E3b: Matching high-div code seeds 3, 4 (needed for cross-domain pairing) ----
log ""
log "--- E3b: Matching high-div code seeds 3, 4 ---"
for seed in 3 4; do
    train_model "code" "5e-4" "$seed"
done

# ===========================================================================
# Phase 3: LMC scans for newly trained models (~3 GPU-hours)
# ===========================================================================
log ""
log "=== Phase 3: LMC scans for new models (~3 GPU-hours) ==="

# ---- Math ↔ General cross-domain (3 seeds) ----
log ""
log "--- Math ↔ General cross-domain LMC ---"
for seed in 0 1 2; do
    out="$RESULTS/lmc_math_general_s${seed}.json"
    if [ -f "$out" ]; then log "  [skip] $out"; continue; fi
    run_lmc_scan "math_lr1e-4_s${seed}" "general_lr1e-4_s${seed}" "$out" "lmc_math_general_s${seed}"
done

# ---- Math within-domain (3 pairs) ----
log ""
log "--- Math within-domain (3 pairs) ---"
for pair in "0 1" "0 2" "1 2"; do
    s1=$(echo $pair | cut -d' ' -f1); s2=$(echo $pair | cut -d' ' -f2)
    out="$RESULTS/lmc_math_s${s1}_s${s2}.json"
    if [ -f "$out" ]; then log "  [skip] $out"; continue; fi
    dir1="$MODELS/math_lr1e-4_s${s1}"; dir2="$MODELS/math_lr1e-4_s${s2}"
    rm -rf "$MODELS/_a" "$MODELS/_b"
    mkdir -p "$MODELS/_a" "$MODELS/_b"
    cp "$dir1/W_math_final.pt" "$MODELS/_a/W_code_final.pt"
    cp "$dir1/W_math_head_final.pt" "$MODELS/_a/W_code_head_final.pt" 2>/dev/null || true
    cp "$dir2/W_math_final.pt" "$MODELS/_b/W_medical_final.pt"
    cp "$dir2/W_math_head_final.pt" "$MODELS/_b/W_medical_head_final.pt" 2>/dev/null || true
    run_lmc_scan "_a" "_b" "$out" "lmc_math_s${s1}_s${s2}"
done

# ---- General within-domain (3 pairs) ----
log ""
log "--- General within-domain (3 pairs) ---"
for pair in "0 1" "0 2" "1 2"; do
    s1=$(echo $pair | cut -d' ' -f1); s2=$(echo $pair | cut -d' ' -f2)
    out="$RESULTS/lmc_general_s${s1}_s${s2}.json"
    if [ -f "$out" ]; then log "  [skip] $out"; continue; fi
    dir1="$MODELS/general_lr1e-4_s${s1}"; dir2="$MODELS/general_lr1e-4_s${s2}"
    rm -rf "$MODELS/_a" "$MODELS/_b"
    mkdir -p "$MODELS/_a" "$MODELS/_b"
    cp "$dir1/W_general_final.pt" "$MODELS/_a/W_code_final.pt"
    cp "$dir1/W_general_head_final.pt" "$MODELS/_a/W_code_head_final.pt" 2>/dev/null || true
    cp "$dir2/W_general_final.pt" "$MODELS/_b/W_medical_final.pt"
    cp "$dir2/W_general_head_final.pt" "$MODELS/_b/W_medical_head_final.pt" 2>/dev/null || true
    run_lmc_scan "_a" "_b" "$out" "lmc_general_s${s1}_s${s2}"
done

# ---- New high-div medical seeds cross-domain scans (seeds 3, 4) ----
log ""
log "--- High-div cross-domain seeds 3, 4 ---"
for seed in 3 4; do
    out="$RESULTS/lmc_lr5e-4_s${seed}.json"
    if [ -f "$out" ]; then log "  [skip] $out"; continue; fi
    run_lmc_scan "code_lr5e-4_s${seed}" "medical_lr5e-4_s${seed}" "$out" "lmc_lr5e-4_s${seed}"
done

# ---- High-div medical within-domain with new seeds (additional pairs) ----
log ""
log "--- High-div medical within-domain (with new seeds) ---"
for pair in "0 3" "0 4" "1 3" "1 4" "2 3" "2 4" "3 4"; do
    s1=$(echo $pair | cut -d' ' -f1); s2=$(echo $pair | cut -d' ' -f2)
    out="$RESULTS/lmc_medical_high_s${s1}_s${s2}.json"
    if [ -f "$out" ]; then log "  [skip] $out"; continue; fi
    dir1="$MODELS/medical_lr5e-4_s${s1}"; dir2="$MODELS/medical_lr5e-4_s${s2}"
    rm -rf "$MODELS/_a" "$MODELS/_b"
    mkdir -p "$MODELS/_a" "$MODELS/_b"
    cp "$dir1/W_medical_final.pt" "$MODELS/_a/W_code_final.pt"
    cp "$dir1/W_medical_head_final.pt" "$MODELS/_a/W_code_head_final.pt" 2>/dev/null || true
    cp "$dir2/W_medical_final.pt" "$MODELS/_b/W_medical_final.pt"
    cp "$dir2/W_medical_head_final.pt" "$MODELS/_b/W_medical_head_final.pt" 2>/dev/null || true
    run_lmc_scan "_a" "_b" "$out" "lmc_medical_high_s${s1}_s${s2}"
done

# ===========================================================================
# Cleanup
# ===========================================================================
rm -rf "$MODELS/_a" "$MODELS/_b" "$MODELS/_p1" "$MODELS/_p2"
rm -f "$MODELS/code_e1" "$MODELS/medical_e1" "$RESULTS/lmc_barrier_c1m1.json"

echo ""
echo "============================================"
log "Phase 1 Batch COMPLETE"
echo "============================================"
