#!/bin/bash
# ===========================================================================
# AFP Phase 0 Pipeline — IVN → F-IVN → Report → Commit
# Fully autonomous, survives terminal disconnect.
# ===========================================================================
set -euo pipefail
PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT"
VENV=/home/jiayu/FCL-PRM-cdspi/venv/bin/python3
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$PIPELINE_LOG"; }
PIPELINE_LOG="$PROJECT/experiments/pipeline_$(date +%Y%m%d_%H%M%S).log"

log "========================================="
log "AFP Phase 0 Pipeline Start"
log "========================================="

# ==============================
# Step 1: IVN (weight-space)
# ==============================
log "Step 1: IVN Phase 0"
IVN_LOG="$PROJECT/experiments/phase0_ivn/logs/run_$(date +%Y%m%d_%H%M%S).log"
log "  log: $IVN_LOG"

$VENV -u scripts/run_ivn_phase0.py \
    --importance mas --gate rational --tau 0.5 --max-rounds 30 \
    >> "$IVN_LOG" 2>&1
IVN_RC=$?

if [ $IVN_RC -eq 0 ] && [ -f "$PROJECT/experiments/phase0_ivn/ivn_results.json" ]; then
    log "IVN PASS"
    # Extract key results for quick view
    $VENV -c "
import json
r = json.load(open('$PROJECT/experiments/phase0_ivn/ivn_results.json'))
print(f'  cosine={r[\"importance_cosine\"]:.3f}')
print(f'  IVN net={r[\"ivn\"][\"net\"]:+.4f} rounds={r[\"ivn\"][\"rounds\"]}')
print(f'  AFP net={r[\"afp_oneshot\"][\"net\"]:+.4f} tau={r[\"afp_oneshot\"][\"tau\"]}')
print(f'  FedAvg net={r[\"fedavg\"][\"net\"]:+.4f} alpha={r[\"fedavg\"][\"alpha\"]}')
" 2>&1 | tee -a "$PIPELINE_LOG"
else
    log "IVN FAILED (rc=$IVN_RC)"
    cat "$PROJECT/experiments/phase0_ivn/crashes/"*.json 2>/dev/null | tee -a "$PIPELINE_LOG"
fi

# ==============================
# Step 2: F-IVN (function-space)
# ==============================
log ""
log "Step 2: F-IVN Phase 0"
FIVN_LOG="$PROJECT/experiments/phase0_fivn/logs/run_$(date +%Y%m%d_%H%M%S).log"
log "  log: $FIVN_LOG"

$VENV -u scripts/run_fivn_phase0.py \
    --tau 0.5 --max-rounds 30 \
    >> "$FIVN_LOG" 2>&1
FIVN_RC=$?

if [ $FIVN_RC -eq 0 ] && [ -f "$PROJECT/experiments/phase0_fivn/fivn_results.json" ]; then
    log "F-IVN PASS"
    $VENV -c "
import json
r = json.load(open('$PROJECT/experiments/phase0_fivn/fivn_results.json'))
print(f'  F-IVN net={r[\"fivn\"][\"net\"]:+.4f} rounds={r[\"fivn\"][\"rounds\"]}')
" 2>&1 | tee -a "$PIPELINE_LOG"
else
    log "F-IVN FAILED (rc=$FIVN_RC)"
    cat "$PROJECT/experiments/phase0_fivn/crashes/"*.json 2>/dev/null | tee -a "$PIPELINE_LOG"
fi

# ==============================
# Step 3: Generate Report
# ==============================
log ""
log "Step 3: Generate HTML Report"

$VENV scripts/generate_phase0_report.py 2>&1 | tee -a "$PIPELINE_LOG"
# Also update the static docs
$VENV scripts/generate_reports.py 2>&1 | tee -a "$PIPELINE_LOG"

REPORT_PATH=$(ls -t docs/reports/phase0-results-*.html 2>/dev/null | head -1)
log "Report: $REPORT_PATH"

# ==============================
# Step 4: Git Commit
# ==============================
log ""
log "Step 4: Git Commit"

git add experiments/phase0_ivn/ivn_results.json \
        experiments/phase0_fivn/fivn_results.json \
        docs/reports/phase0-results-*.html \
        docs/reports/*.html \
        experiments/pipeline_*.log \
        2>/dev/null || true

git commit -m "feat(phase0): IVN + F-IVN experiment results

IVN: Qwen2.5-Coder ⇄ Qwen2.5-Math, weight-space negotiation
F-IVN: Qwen2.5-Math ⇄ Pythia-1.4B, function-space negotiation

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>" 2>&1 | tee -a "$PIPELINE_LOG" || log "git commit skipped (no changes)"

# ==============================
# Done
# ==============================
log ""
log "========================================="
log "Pipeline Complete"
log "========================================="
log "Results: experiments/phase0_ivn/ivn_results.json"
log "         experiments/phase0_fivn/fivn_results.json"
log "Report:  $REPORT_PATH"
log "Log:     $PIPELINE_LOG"
