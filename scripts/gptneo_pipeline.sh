#!/bin/bash
# GPT-Neo-1.3B pipeline — streaming tokenize + train + LMC
# Fix: single JSONL pass, batch_encode, no list-of-tensors, no double-train
set -uo pipefail; cd /home/jiayu/AFP
export HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
V=/home/jiayu/AFP/venv/bin/python3; TS=$(date +%Y%m%d_%H%M%S); MID=EleutherAI/gpt-neo-1.3B
M=/home/jiayu/AFP/experiments/trained_models_gptneo
R=/home/jiayu/AFP/experiments/phase0_gptneo/results
L=/home/jiayu/AFP/experiments/phase0_gptneo
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "=== Data caches (streaming, single JSONL pass) ==="
$V -c "
import torch,json,gc; from transformers import AutoTokenizer; from pathlib import Path
tok=AutoTokenizer.from_pretrained('$MID',local_files_only=True)
if tok.pad_token is None: tok.pad_token=tok.eos_token
P=Path('.'); MAXL=256; BATCH=512
PAD=tok.pad_token_id

# ── Pass 1: stream JSONL once, bin raw (txt,label) per domain ──
RAW = {'code': [], 'medical': []}
with open(P/'data'/'versaprm'/'versa_prm.jsonl') as f:
    for line in f:
        d = json.loads(line)
        dom = d.get('domain', 'general')
        if dom not in RAW: continue
        q = d.get('question', '')
        # iterate multi-step structure — same as train_agent.py (plural 'steps')
        for step_txt, lbl in zip(d.get('steps', []), d.get('labels', [])):
            RAW[dom].append((f'{q}\n{step_txt}', 1.0 if int(lbl) == 1 else 0.0))
del d; gc.collect()
print(f'Streamed JSONL → code={len(RAW[\"code\"])} medical={len(RAW[\"medical\"])} steps')

for dom in ['code', 'medical']:
    pairs = RAW[dom]
    n = len(pairs)
    # Shuffle once for this domain
    perm = torch.randperm(n).tolist()
    n_val = min(2000, int(n * 0.15))
    n_train = n - n_val
    splits = {'train': (perm[:n_train], n_train), 'val': (perm[n_train:], n_val)}

    for dtype, (idxs, total) in splits.items():
        cache = P/'data'/'versaprm'/f'{dtype}_{dom}_gptneo_L{MAXL}.pt'
        if cache.exists():
            print(f'  [skip] {dtype} {dom} cache exists')
            continue
        # Pre-allocate ONE output tensor — never accumulate tiny tensors
        inp  = torch.full((total, MAXL), PAD, dtype=torch.long)
        msk  = torch.zeros(total, MAXL, dtype=torch.bool)
        labs = torch.zeros(total, dtype=torch.float32)
        kept = 0

        for i in range(0, total, BATCH):
            b_idxs  = idxs[i:i+BATCH]
            b_texts = [pairs[j][0] for j in b_idxs]
            b_labs  = [pairs[j][1] for j in b_idxs]
            enc = tok.batch_encode_plus(
                b_texts, truncation=True, max_length=MAXL,
                padding='max_length', return_tensors=None)  # ← Python lists, NOT tensors
            for k in range(len(b_texts)):
                ids = enc['input_ids'][k]
                L = min(len(ids), MAXL)
                inp[kept, :L] = torch.tensor(ids[:L], dtype=torch.long)
                msk[kept, :L] = True
                labs[kept] = b_labs[k]
                kept += 1

        torch.save({'input_ids': inp[:kept], 'attention_mask': msk[:kept],
                    'labels': labs[:kept], 'n': kept}, cache)
        print(f'  {dtype} {dom}: {kept} samples saved')

    # Free domain data before next domain (peak memory = 1 domain at a time)
    del RAW[dom]; gc.collect()

print('Caches done (streaming)')
" >> "$L/caches_$TS.log" 2>&1

# Swap training data for GPT-Neo tokenizer
# (train_agent.py always reads _pythia_ cache paths — we overwrite them)
cp data/versaprm/train_code_gptneo_L256.pt data/versaprm/train_code_pythia_L256.pt
cp data/versaprm/val_code_gptneo_L256.pt data/versaprm/val_code_pythia_L256.pt
cp data/versaprm/train_medical_gptneo_L256.pt data/versaprm/train_medical_pythia_L256.pt
cp data/versaprm/val_medical_gptneo_L256.pt data/versaprm/val_medical_pythia_L256.pt

log "=== Training (6 models, 2-epoch) ==="
for domain in code medical; do for seed in 0 1 2; do
    outdir="$M/${domain}_lr1e-4_s${seed}"
    [ -f "$outdir/W_${domain}_final.pt" ] && { log "  [skip] $domain s$seed"; continue; }
    mkdir -p "$outdir"; log "  Training $domain s$seed..."
    $V -u scripts/train_agent.py --domain "$domain" --lr 1e-4 --epochs 2 \
        --output-dir "$outdir" --model-id "$MID" \
        >> "$L/train_${domain}_s${seed}_$TS.log" 2>&1
    $V -c "import torch; torch.cuda.empty_cache()" 2>/dev/null
done; done

log "=== Cross-domain LMC ==="
for s in 0 1 2; do
    out="$R/lmc_gptneo_cross_s${s}.json"; [ -f "$out" ] && continue
    $V -u scripts/lmc_3pt_scan.py \
        --model-a "$M/code_lr1e-4_s${s}" --model-b "$M/medical_lr1e-4_s${s}" \
        --domain-a code --domain-b medical --output "$out" \
        >> "$L/lmc_cross_s${s}_$TS.log" 2>&1
done

log "=== Within-domain LMC ==="
for domain in code medical; do for p in "0 1" "0 2" "1 2"; do
    s1=$(echo $p|cut -d" " -f1); s2=$(echo $p|cut -d" " -f2)
    out="$R/lmc_gptneo_${domain}_within_s${s1}_s${s2}.json"; [ -f "$out" ] && continue
    $V -u scripts/lmc_3pt_scan.py \
        --model-a "$M/${domain}_lr1e-4_s${s1}" --model-b "$M/${domain}_lr1e-4_s${s2}" \
        --domain-a "$domain" --domain-b "$domain" --output "$out" \
        >> "$L/lmc_${domain}_s${s1}_s${s2}_$TS.log" 2>&1
done; done

log "=== GPT-Neo DONE ==="
