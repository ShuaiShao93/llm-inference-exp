#!/usr/bin/env bash
# Name the kernel that actually faults, and tighten the overflow boundary.
# CUDA_LAUNCH_BLOCKING=1 makes the reported Python frame the real launch site;
# without it the error surfaces at an unrelated later synchronize().
set -u

PY=/usr/bin/python3.12
LORA=${LORA:-$HOME/model_ckpt/synthetic-loras/gemma-4-e2b-r16-stripped}
OUT=${OUT:-$HOME/e2b_ima_results}
mkdir -p "$OUT"

echo "=== launch-blocking run at 87700 (known crashing) ==="
CUDA_LAUNCH_BLOCKING=1 timeout 1800 "$PY" scripts/repro_e2b_lora_ima.py \
  --max_model_len 87700 --lora "$LORA" > "$OUT/clb87700.log" 2>&1
echo "rc=$?"

echo "=== tighten boundary around the predicted 87381 ==="
for n in 87300 87450; do
  timeout 1200 "$PY" scripts/repro_e2b_lora_ima.py \
    --max_model_len "$n" --lora "$LORA" > "$OUT/p$n.log" 2>&1
  rc=$?
  if grep -q 'ENGINE INIT OK' "$OUT/p$n.log"; then v=PASS
  elif grep -q 'illegal memory access' "$OUT/p$n.log"; then v='CRASH illegal-memory-access'
  elif [ $rc -eq 124 ]; then v=TIMEOUT
  else v="OTHER rc=$rc"; fi
  printf '%-10s %-8s %s\n' "p$n" "$n" "$v" | tee -a "$OUT/summary.txt"
done
echo PINPOINT_DONE
