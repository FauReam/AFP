#!/bin/bash
# GPT-Neo-1.3B pipeline — streaming tokenize (mem-capped) + train + LMC
# Memory budget: hard cap per domain prevents OOM on 121GB unified memory
set -uo pipefail; cd /home/jiayu/AFP
export HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
V=/home/jiayu/AFP/venv/bin/python3; TS=$(date +%Y%m%d_%H%M%S); MID=EleutherAI/gpt-neo-1.3B
M=/home/jiayu/AFP/experiments/trained_models_gptneo
R=/home/jiayu/AFP/experiments/phase0_gptneo/results
L=/home/jiayu/AFP/experiments/phase0_gptneo
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "=== Data caches (streaming, single JSONL pass, mem-capped) ==="
$V -c "
import torch,json,gc,os; from transformers import AutoTokenizer; from pathlib import Path
tok=AutoTokenizer.from_pretrained('$MID',local_files_only=True)
if tok.pad_token is None: tok.pad_token=tok.eos_token
P=Path('.'); MAXL=256; BATCH=512; PAD=tok.pad_token_id
BYTES_PER_STEP = MAXL*8 + MAXL*1 + 4  # int64 ids + bool mask + float32 label ≈ 2.3KB
MAX_STEPS_PER_DOMAIN = 8_000_000        # ~18.4 GB tensor — leaves margin under 80GB
GB_PER_STEP = BYTES_PER_STEP / 1e9
TARGET_GB = 80

def mem_avail():
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if 'MemAvailable' in line:
                    return int(line.split()[1]) / 1e6
    except: pass
    return -1

def check_mem(label):
    avail = mem_avail()
    if 0 < avail < 10:
        print(f'  !! LOW MEM [{label}]: MemAvailable={avail:.1f} GB !!')
    else:
        print(f'  [{label}] MemAvailable={avail:.1f} GB')
    return avail

check_mem('start')

# ── Scan: stream JSONL once, bin (text,label) per domain ──
RAW = {'code': [], 'medical': []}
with open(P/'data'/'versaprm'/'versa_prm.jsonl') as f:
    for line in f:
        d = json.loads(line)
        dom = d.get('domain', 'general')
        if dom not in RAW: continue
        q = d.get('question', '')
        for step_txt, lbl in zip(d.get('steps', []), d.get('labels', [])):
            RAW[dom].append((f'{q}\n{step_txt}', 1.0 if int(lbl) == 1 else 0.0))
del d; gc.collect()

for dom in ['code', 'medical']:
    n_raw = len(RAW[dom])
    print(f'  {dom}: {n_raw} steps (~{n_raw*GB_PER_STEP:.1f} GB tensor)')
check_mem('after scan')

for dom in ['code', 'medical']:
    pairs = RAW[dom]
    n = len(pairs)

    # ── Hard cap per domain ──
    if n > MAX_STEPS_PER_DOMAIN:
        print(f'  !! CAP {dom}: {n:,} → {MAX_STEPS_PER_DOMAIN:,} steps (tensor would be {n*GB_PER_STEP:.1f} GB > safe limit)')
        import random; random.seed(42)
        keep = set(sorted(random.sample(range(n), MAX_STEPS_PER_DOMAIN)))
        pairs = [p for i, p in enumerate(pairs) if i in keep]
        n = MAX_STEPS_PER_DOMAIN

    # ── Mem check before allocation ──
    est_tensor_gb = n * GB_PER_STEP
    avail = check_mem(f'before {dom} tensor')
    if 0 < avail < est_tensor_gb + 5:
        print(f'  !! ABORT {dom}: need {est_tensor_gb:.1f} GB but only {avail:.1f} GB available')
        continue

    # Shuffle
    perm = torch.randperm(n).tolist()
    n_val = min(2000, int(n * 0.15))
    n_train = n - n_val
    splits = {'train': (perm[:n_train], n_train), 'val': (perm[n_train:], n_val)}

    for dtype, (idxs, total) in splits.items():
        cache = P/'data'/'versaprm'/f'{dtype}_{dom}_gptneo_L{MAXL}.pt'
        if cache.exists():
            print(f'  [skip] {dtype} {dom} cache exists')
            continue

        print(f'  tokenizing {dtype} {dom}: {total:,} steps ({total*GB_PER_STEP:.1f} GB)')
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
                padding='max_length', return_tensors=None)
            for k in range(len(b_texts)):
                ids = enc['input_ids'][k]
                L = min(len(ids), MAXL)
                inp[kept, :L] = torch.tensor(ids[:L], dtype=torch.long)
                msk[kept, :L] = True
                labs[kept] = b_labs[k]
                kept += 1

        torch.save({'input_ids': inp[:kept], 'attention_mask': msk[:kept],
                    'labels': labs[:kept], 'n': kept}, cache)
        print(f'  [saved] {dtype} {dom}: {kept} samples')
        del inp, msk, labs; gc.collect()

    del RAW[dom]; gc.collect()
    check_mem(f'after {dom}')

print(f'Caches done. MemAvailable={mem_avail():.1f} GB')
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
