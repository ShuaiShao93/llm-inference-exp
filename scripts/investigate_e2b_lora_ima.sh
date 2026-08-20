#!/usr/bin/env bash
# Isolate the Gemma 4 E2B + LoRA illegal-memory-access at long context.
# Stage 1: find the smallest input length that crashes (10k is known-good).
# Stage 2: no-LoRA control at 100k.
# Each run is one process; never two at once (shared GPU).
set -u

PY=/usr/bin/python3.12
MODEL=${MODEL:-Neural-ICE/Gemma-4-E2B-it-NVFP4}
PREC=${PREC:-fp4}
LORA=${LORA:-$HOME/model_ckpt/synthetic-loras/gemma-4-e2b-r16-stripped}
BACKEND=${BACKEND:-FLASHINFER}
OUT=${OUT:-/tmp/e2b_ima}
mkdir -p "$OUT"

run() {
  local tag="$1"; shift
  local log="$OUT/$tag.log"
  timeout 900 $PY scripts/vllm_local.py --num_runs 1 "$@" > "$log" 2>&1
  local rc=$?
  local verdict
  if grep -q "Mean latency" "$log"; then
    verdict="PASS $(grep 'Mean latency' "$log" | awk '{print $3, $4}')"
  elif grep -q "an illegal memory access was encountered" "$log"; then
    verdict="CRASH illegal-memory-access"
  elif grep -q "CUBLAS_STATUS_EXECUTION_FAILED" "$log"; then
    verdict="CRASH cublas-exec-failed"
  elif grep -q "OutOfMemoryError" "$log"; then
    verdict="OOM"
  elif [ $rc -eq 124 ]; then
    verdict="TIMEOUT"
  else
    verdict="OTHER rc=$rc"
  fi
  printf '%-40s %s\n' "$tag" "$verdict"
}

echo "=== stage 1: length threshold (with LoRA, $BACKEND) ==="
for N in 12000 16000 24000 32000 48000 64000; do
  run "len${N}_lora" --model "$MODEL" --precision "$PREC" --input_tokens "$N" \
      --attention_backend "$BACKEND" --lora "$LORA"
done

echo "=== stage 2: no-LoRA control ==="
for N in 100000; do
  run "len${N}_nolora" --model "$MODEL" --precision "$PREC" --input_tokens "$N" \
      --attention_backend "$BACKEND"
done

echo "=== DONE ==="
