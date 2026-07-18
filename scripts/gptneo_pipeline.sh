#!/bin/bash
set -uo pipefail; cd /home/jiayu/AFP
export HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
V=/home/jiayu/AFP/venv/bin/python3; TS=$(date +%Y%m%d_%H%M%S); MID=EleutherAI/gpt-neo-1.3B
M=/home/jiayu/AFP/experiments/trained_models_gptneo
R=/home/jiayu/AFP/experiments/phase0_gptneo/results
L=/home/jiayu/AFP/experiments/phase0_gptneo
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "=== Data caches ==="
$V -c "
import torch,json; from transformers import AutoTokenizer; from pathlib import Path
tok=AutoTokenizer.from_pretrained('$MID',local_files_only=True)
if tok.pad_token is None: tok.pad_token=tok.eos_token
P=Path('.'); MAXL=256
for dom in ['code','medical']:
    for dtype in ['train','val']:
        cache=P/'data'/'versaprm'/f'{dtype}_{dom}_gptneo_L{MAXL}.pt'
        if cache.exists(): continue
        samples=[]
        with open(P/'data'/'versaprm'/'versa_prm.jsonl') as f:
            for line in f:
                d=json.loads(line)
                if d.get('domain','general')!=dom: continue
                samples.append(d)
        split=int(len(samples)*0.8)
        subset=samples[:split] if dtype=='train' else samples[split:split+2000]
        ids,mask,labels=[],[],[]
        for s in subset:
            txt=s.get('question','')+' '+s.get('step','')
            enc=tok(txt,truncation=True,max_length=MAXL,padding='max_length',return_tensors='pt')
            ids.append(enc['input_ids'][0]); mask.append(enc['attention_mask'][0])
            labels.append(1.0 if s.get('label',0)>0 else 0.0)
        torch.save({'input_ids':torch.stack(ids),'attention_mask':torch.stack(mask),'labels':torch.tensor(labels),'n':len(labels)},cache)
        print(f'{dtype} {dom}: {len(labels)} samples')
print('Caches done')
" >> "$L/caches_$TS.log" 2>&1

# Swap training data for GPT-Neo tokenizer (restore after training)
cp data/versaprm/train_code_gptneo_L256.pt data/versaprm/train_code_pythia_L256.pt
cp data/versaprm/val_code_gptneo_L256.pt data/versaprm/val_code_pythia_L256.pt
cp data/versaprm/train_medical_gptneo_L256.pt data/versaprm/train_medical_pythia_L256.pt
cp data/versaprm/val_medical_gptneo_L256.pt data/versaprm/val_medical_pythia_L256.pt

log "=== Training (6 models, 2-epoch) ==="
for domain in code medical; do for seed in 0 1 2; do
    outdir="$M/${domain}_lr1e-4_s${seed}"
    [ -f "$outdir/W_${domain}_final.pt" ] && { log "  [skip] $domain s$seed"; continue; }
    mkdir -p "$outdir"; log "  Training $domain s$seed..."; $V -u scripts/train_agent.py --domain "$domain" --lr 1e-4 --epochs 2 --output-dir "$outdir" --model-id "$MID" >> "$L/train_${domain}_s${seed}_$TS.log" 2>&1; $V -c "import torch; torch.cuda.empty_cache()"
    ($V -u scripts/train_agent.py --domain "$domain" --lr 1e-4 --epochs 2 --output-dir "$outdir" --model-id "$MID" ) >> "$L/train_${domain}_s${seed}_$TS.log" 2>&1
done; done

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

log "=== GPT-Neo DONE ==="
