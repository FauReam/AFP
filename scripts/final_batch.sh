#!/bin/bash
# Final batch: TIES merge + OPT trajectory/Gaussian/4-domain + Pythia layer-selective
# ~22 GPU-hours, timeout-protected, offline-safe
set -uo pipefail; cd /home/jiayu/AFP
export HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
V=/home/jiayu/AFP/venv/bin/python3; TS=$(date +%Y%m%d_%H%M%S)
LO=/home/jiayu/AFP/experiments/phase0_opt
LP=/home/jiayu/AFP/experiments/phase0_training
log() { echo "[$(date +%H:%M:%S)] $*"; }

run_to() {
    local label="$1" timeout_s="$2"; shift 2
    log "  [$label] (timeout=${timeout_s}s)"
    if timeout "$timeout_s" "$@" >> "${LP}/final_${label}_$TS.log" 2>&1; then
        log "  [$label] OK"; return 0
    else
        local rc=$?; [ $rc -eq 124 ] && log "  [$label] TIMEOUT" || log "  [$label] FAILED (exit=$rc)"
        return 1
    fi
}

# ===================================================================
log "============================================"
log "FINAL BATCH: TIES + OPT + Layer-Selective"
log "============================================"

# ---- 1. TIES Merge Benchmark (Pythia) ~2h ----
log ""
log "=== 1. TIES Merge Benchmark (Pythia) ~2h ==="
run_to "ties_merge" 7200 $V -u /home/jiayu/AFP/scripts/merge_benchmark.py
[ -f /home/jiayu/AFP/experiments/phase0_ivn/results/merge_benchmark.json ] && log "  Merge results saved"

# ---- 2. OPT Trajectory (code+medical, 40-step checkpoints) ~4.5h ----
log ""
log "=== 2. OPT Trajectory ~4.5h ==="
OPT_MOD=/home/jiayu/AFP/experiments/trained_models_opt
OPT_RES=/home/jiayu/AFP/experiments/phase0_opt/results
TRAJ_STEP=40

for domain in code medical; do
    outdir="$OPT_MOD/${domain}_trajectory"
    if [ -f "$outdir/W_${domain}_final.pt" ]; then
        log "  [skip] $domain trajectory training"
        continue
    fi
    mkdir -p "$outdir"
    log "  Training $domain with trajectory checkpoints..."
    run_to "opt_traj_train_${domain}" 9000 \
        $V -u /home/jiayu/AFP/scripts/opt/train_agent.py --domain "$domain" --lr 1e-4 --epochs 2 \
            --output-dir "$outdir" --trajectory-step "$TRAJ_STEP"
done

log "  LMC scans for OPT trajectory..."
BASE_OPT_PT="$OPT_MOD/_opt_base.pt"
if [ ! -f "$BASE_OPT_PT" ]; then
    $V -c "
import torch; from transformers import AutoModel
m = AutoModel.from_pretrained('facebook/opt-1.3b',local_files_only=True,torch_dtype=torch.bfloat16)
torch.save({k:v.detach().cpu() for k,v in m.state_dict().items()}, '$BASE_OPT_PT')
print('OPT base saved')
"
fi

for domain in code medical; do
    trajdir="$OPT_MOD/${domain}_trajectory/trajectory"
    for ckpt in $(ls "$trajdir"/step_[0-9]*.pt 2>/dev/null | grep -v head | sort -t_ -k2 -n); do
        step=$(basename "$ckpt" .pt | sed 's/step_//')
        out="$OPT_RES/lmc_traj_opt_${domain}_step${step}.json"
        [ -f "$out" ] && continue
        run_to "opt_traj_${domain}_s${step}" 1800 \
            $V -u /home/jiayu/AFP/scripts/opt/lmc_3pt_scan.py \
                --model-a "$BASE_OPT_PT" --model-b "$ckpt" --domain-a opt --domain-b opt --output "$out"
    done
done

# ---- 3. OPT Gaussian Calibration (5 levels) ~2h ----
log ""
log "=== 3. OPT Gaussian Calibration ~2h ==="
for dw in 0.5 1.0 2.0 4.0 8.0; do
    out="$OPT_RES/noise_gaussian_opt_dw${dw}.json"
    [ -f "$out" ] && { log "  [skip] ΔW=${dw}%"; continue; }
    log "  Gaussian ΔW≈${dw}%"
    $V -c "
import torch, os; from transformers import AutoModel
base = AutoModel.from_pretrained('facebook/opt-1.3b',local_files_only=True,torch_dtype=torch.bfloat16)
bs = {k:v.detach().cpu() for k,v in base.state_dict().items()}
bn = sum(v.float().norm().item()**2 for v in bs.values())**0.5; del base
torch.manual_seed(42)
noise = {}
for k,v in bs.items():
    noise[k] = v.float() + torch.randn_like(v.float())*v.float().std()
cd = sum((noise[k]-bs[k]).float().norm().item()**2 for k in bs)**0.5
scale = $dw / (cd/(bn+1e-8)*100)
for k in noise: noise[k] = (bs[k].float()+(noise[k]-bs[k].float())*scale).to(bs[k].dtype)
os.makedirs('$OPT_MOD/_a',exist_ok=True); os.makedirs('$OPT_MOD/_b',exist_ok=True)
torch.save(bs, '$OPT_MOD/_a/W_code_final.pt'); torch.save(noise, '$OPT_MOD/_b/W_medical_final.pt')
import shutil; shutil.copy('$OPT_MOD/code_lr1e-4_s0/W_code_head_final.pt','$OPT_MOD/_a/W_code_head_final.pt')
shutil.copy('$OPT_MOD/medical_lr1e-4_s0/W_medical_head_final.pt','$OPT_MOD/_b/W_medical_head_final.pt')
" >> "${LO}/gauss_opt_${dw}_$TS.log" 2>&1
    run_to "gauss_opt_dw${dw}" 1200 \
        $V -u /home/jiayu/AFP/scripts/opt/lmc_3pt_scan.py \
            --model-a "$OPT_MOD/_a" --model-b "$OPT_MOD/_b" --domain-a code --domain-b medical --output "$out"
done

# ---- 4. OPT general within + per-block fix ~1h ----
log ""
log "=== 4. OPT general within + per-block ~1h ==="
for p in "0 1" "0 2" "1 2"; do
    s1=$(echo $p|cut -d' ' -f1); s2=$(echo $p|cut -d' ' -f2)
    out="$OPT_RES/lmc_general_within_opt2ep_s${s1}_s${s2}.json"
    [ -f "$out" ] && { log "  [skip] general_s${s1}_s${s2}"; continue; }
    run_to "gen_within_s${s1}_s${s2}" 1200 \
        $V -u /home/jiayu/AFP/scripts/opt/lmc_3pt_scan.py \
            --model-a "$OPT_MOD/general_lr1e-4_s${s1}" --model-b "$OPT_MOD/general_lr1e-4_s${s2}" \
            --domain-a general --domain-b general --output "$out"
done
# Per-block divergence for OPT
run_to "opt_per_block" 600 $V -c "
import torch, json; from transformers import AutoModel
base = AutoModel.from_pretrained('facebook/opt-1.3b',local_files_only=True,torch_dtype=torch.bfloat16)
bs = {k:v.detach().cpu() for k,v in base.state_dict().items()}; del base
res = {}
for domain in ['code','medical']:
    sd = torch.load('$OPT_MOD/${domain}_lr1e-4_s0/W_${domain}_final.pt',map_location='cpu',weights_only=True)
    divs = []
    for blk in range(24):
        bk = [k for k in sd if f'layers.{blk}.' in k]
        dw = sum((sd[k]-bs[k]).float().norm().item()**2 for k in bk if k in bs)**0.5
        n = sum(bs[k].float().norm().item()**2 for k in bk if k in bs)**0.5
        divs.append(round(dw/(n+1e-8)*100,3))
    res[domain] = {'divs':divs,'mean':sum(divs)/len(divs)}
import numpy as np; r = np.corrcoef(res['code']['divs'],res['medical']['divs'])[0,1]
res['pearson_r'] = round(float(r),4)
with open('$OPT_RES/per_block_divergence.json','w') as f: json.dump(res,f,indent=2)
print(f'OPT per-block r={r:.4f}')
"

# ---- 5. OPT math+general (3 seeds, 2-epoch) + LMC ~9h ----
log ""
log "=== 5. OPT math+general (3 seeds) + LMC ~9h ==="
for domain in math general; do
    for seed in 0 1 2; do
        outdir="$OPT_MOD/${domain}_lr1e-4_s${seed}"
        [ -f "$outdir/W_${domain}_final.pt" ] && { log "  [skip] $domain s$seed"; continue; }
        mkdir -p "$outdir"
        log "  Training $domain s$seed (2 epochs)..."
        run_to "opt_train_${domain}_s${seed}" 9000 \
            $V -u /home/jiayu/AFP/scripts/opt/train_agent.py --domain "$domain" --lr 1e-4 --epochs 2 --output-dir "$outdir"
    done
done
# LMC: math↔general cross + within
for seed in 0 1 2; do
    out="$OPT_RES/lmc_math_gen_opt2ep_s${seed}.json"
    [ -f "$out" ] && continue
    run_to "opt_mg_cross_s${seed}" 1200 \
        $V -u /home/jiayu/AFP/scripts/opt/lmc_3pt_scan.py \
            --model-a "$OPT_MOD/math_lr1e-4_s${seed}" --model-b "$OPT_MOD/general_lr1e-4_s${seed}" \
            --domain-a math --domain-b general --output "$out"
done
for domain in math general; do
    for p in "0 1" "0 2" "1 2"; do
        s1=$(echo $p|cut -d' ' -f1); s2=$(echo $p|cut -d' ' -f2)
        out="$OPT_RES/lmc_${domain}_within_opt2ep_s${s1}_s${s2}.json"
        [ -f "$out" ] && continue
        run_to "opt_${domain}_w_s${s1}_s${s2}" 1200 \
            $V -u /home/jiayu/AFP/scripts/opt/lmc_3pt_scan.py \
                --model-a "$OPT_MOD/${domain}_lr1e-4_s${s1}" --model-b "$OPT_MOD/${domain}_lr1e-4_s${s2}" \
                --domain-a "$domain" --domain-b "$domain" --output "$out"
    done
done

# ---- 6. Pythia Layer-Selective Within-Domain Merge ~2h ----
log ""
log "=== 6. Pythia Layer-Selective Within-Domain ~2h ==="
PR=/home/jiayu/AFP/experiments/phase0_ivn/results
PM=/home/jiayu/AFP/experiments/trained_models
for domain in code medical; do
    for layers in "early:0-7" "mid:8-15" "late:16-23" "all:0-23"; do
        label=$(echo $layers|cut -d: -f1)
        lstart=$(echo $layers|cut -d: -f2|cut -d- -f1); lend=$(echo $layers|cut -d: -f2|cut -d- -f2)
        out="$PR/lmc_layers_within_${domain}_${label}.json"
        [ -f "$out" ] && { log "  [skip] $domain $label"; continue; }
        log "  $domain layer-$label within"
        $V -c "
import torch,os
sd_a = torch.load('$PM/${domain}_lr1e-4_s0/W_${domain}_final.pt',map_location='cpu',weights_only=True)
sd_b = torch.load('$PM/${domain}_lr1e-4_s1/W_${domain}_final.pt',map_location='cpu',weights_only=True)
merged = {}
for k in sd_a:
    lm = any(f'layers.{l}.' in k for l in range($lstart,$lend+1))
    if lm and k in sd_b: merged[k] = 0.5*sd_a[k].float() + 0.5*sd_b[k].float()
    else: merged[k] = sd_a[k].float()
os.makedirs('$PM/_a',exist_ok=True); os.makedirs('$PM/_b',exist_ok=True)
torch.save(sd_a,'$PM/_a/W_code_final.pt'); torch.save(merged,'$PM/_b/W_medical_final.pt')
import shutil
shutil.copy('$PM/${domain}_lr1e-4_s0/W_${domain}_head_final.pt','$PM/_a/W_code_head_final.pt')
shutil.copy('$PM/${domain}_lr1e-4_s0/W_${domain}_head_final.pt','$PM/_b/W_medical_head_final.pt')
" >> "${LP}/layers_within_${domain}_${label}_$TS.log" 2>&1
        run_to "layers_${domain}_${label}" 1200 \
            $V -u /home/jiayu/AFP/scripts/lmc_3pt_scan.py \
                --model-a "$PM/_a" --model-b "$PM/_b" --domain-a "$domain" --domain-b "$domain" --output "$out"
    done
done

# Cleanup
rm -rf "$PM/_a" "$PM/_b" "$OPT_MOD/_a" "$OPT_MOD/_b" "$OPT_MOD/code_e1" "$OPT_MOD/medical_e1"

log ""
log "============================================"
log "FINAL BATCH COMPLETE"
log "============================================"
