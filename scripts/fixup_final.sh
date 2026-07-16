#!/bin/bash
# Fixup: OPT trajectory + Pythia layer-selective within + OPT general
set -uo pipefail; cd /home/jiayu/AFP
export HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
V=/home/jiayu/AFP/venv/bin/python3; TS=$(date +%Y%m%d_%H%M%S)
L=/home/jiayu/AFP/experiments/phase0_training
log() { echo "[$(date +%H:%M:%S)] $*"; }

run_to() { local label="$1" t="$2"; shift 2; log "  [$label]"; timeout "$t" "$@" >> "$L/fixup_${label}_$TS.log" 2>&1 && log "  [$label] OK" || log "  [$label] FAILED"; }

# ====== 1. Pythia layer-selective within-domain ======
log "=== 1. Pythia layer-selective within ==="
PM=/home/jiayu/AFP/experiments/trained_models
PR=/home/jiayu/AFP/experiments/phase0_ivn/results
for domain in code medical; do
    for layers in "early:0-7" "mid:8-15" "late:16-23" "all:0-23"; do
        label=$(echo $layers|cut -d: -f1)
        lstart=$(echo $layers|cut -d: -f2|cut -d- -f1); lend=$(echo $layers|cut -d: -f2|cut -d- -f2)
        out="$PR/lmc_layers_within_${domain}_${label}.json"
        [ -f "$out" ] && { log "  [skip] $domain $label"; continue; }
        log "  $domain $label ($lstart-$lend)"
        # Setup: create merged model in temp dirs, then pass directly to scan
        rm -rf "$PM/_t1" "$PM/_t2"; mkdir -p "$PM/_t1" "$PM/_t2"
        $V -c "
import torch,os,shutil
sd_a=torch.load('$PM/${domain}_lr1e-4_s0/W_${domain}_final.pt',map_location='cpu',weights_only=True)
sd_b=torch.load('$PM/${domain}_lr1e-4_s1/W_${domain}_final.pt',map_location='cpu',weights_only=True)
merged={}
for k in sd_a:
    in_layer=any(f'layers.{l}.' in k for l in range($lstart,$lend+1))
    if in_layer and k in sd_b: merged[k]=0.5*sd_a[k].float()+0.5*sd_b[k].float()
    else: merged[k]=sd_a[k].float()
torch.save(sd_a,'$PM/_t1/W_${domain}_final.pt'); torch.save(merged,'$PM/_t2/W_${domain}_final.pt')
shutil.copy('$PM/${domain}_lr1e-4_s0/W_${domain}_head_final.pt','$PM/_t1/W_${domain}_head_final.pt')
shutil.copy('$PM/${domain}_lr1e-4_s0/W_${domain}_head_final.pt','$PM/_t2/W_${domain}_head_final.pt')
" 2>/dev/null
        run_to "layers_${domain}_${label}" 1200 \
            $V -u /home/jiayu/AFP/scripts/lmc_3pt_scan.py \
                --model-a "$PM/_t1" --model-b "$PM/_t2" --domain-a "$domain" --domain-b "$domain" --output "$out"
    done
done
rm -rf "$PM/_t1" "$PM/_t2"

# ====== 2. OPT trajectory ======
log "=== 2. OPT trajectory ==="
OM=/home/jiayu/AFP/experiments/trained_models_opt
OR=/home/jiayu/AFP/experiments/phase0_opt/results
BASE_OPT="$OM/_opt_base.pt"
[ ! -f "$BASE_OPT" ] && $V -c "import torch;from transformers import AutoModel;m=AutoModel.from_pretrained('facebook/opt-1.3b',local_files_only=True,torch_dtype=torch.bfloat16);torch.save({k:v.detach().cpu() for k,v in m.state_dict().items()},'$BASE_OPT');print('saved')"

for domain in code medical; do
    trajdir="$OM/${domain}_trajectory/trajectory"
    for ckpt in $(ls "$trajdir"/step_[0-9]*.pt 2>/dev/null | grep -v head | sort -t_ -k2 -n); do
        step=$(basename "$ckpt" .pt | sed 's/step_//')
        out="$OR/lmc_traj_opt_${domain}_step${step}.json"
        [ -f "$out" ] && continue
        log "  OPT traj $domain step=$step"
        # Setup: _t1=base model, _t2=checkpoint, use code/medical naming convention
        rm -rf "$OM/_t1" "$OM/_t2"; mkdir -p "$OM/_t1" "$OM/_t2"
        cp "$BASE_OPT" "$OM/_t1/W_code_final.pt"
        cp "$ckpt" "$OM/_t2/W_medical_final.pt"
        cp "$OM/code_lr1e-4_s0/W_code_head_final.pt" "$OM/_t1/W_code_head_final.pt" 2>/dev/null || true
        cp "$OM/${domain}_lr1e-4_s0/W_${domain}_head_final.pt" "$OM/_t2/W_medical_head_final.pt" 2>/dev/null || true
        run_to "opt_traj_${domain}_s${step}" 1800 \
            $V -u /home/jiayu/AFP/scripts/opt/lmc_3pt_scan.py \
                --model-a "$OM/_t1" --model-b "$OM/_t2" --domain-a code --domain-b medical --output "$out"
    done
done
rm -rf "$OM/_t1" "$OM/_t2"

# ====== 3. OPT general within (1-epoch models) ======
log "=== 3. OPT general within (1-epoch) ==="
for p in "0 1" "0 2" "1 2"; do
    s1=$(echo $p|cut -d' ' -f1); s2=$(echo $p|cut -d' ' -f2)
    a="$OM/general_lr1e-4_s${s1}"; b="$OM/general_lr1e-4_s${s2}"
    [ ! -f "$a/W_general_final.pt" -o ! -f "$b/W_general_final.pt" ] && { log "  [skip] general missing"; continue; }
    out="$OR/lmc_general_within_opt_s${s1}_s${s2}.json"
    [ -f "$out" ] && { log "  [skip] general_s${s1}_s${s2}"; continue; }
    log "  general s${s1}↔s${s2}"
    run_to "opt_gen_w_s${s1}_s${s2}" 1200 \
        $V -u /home/jiayu/AFP/scripts/opt/lmc_3pt_scan.py \
            --model-a "$a" --model-b "$b" --domain-a general --domain-b general --output "$out"
done

log "=== FIXUP DONE ==="
