#!/usr/bin/env bash
# Bisect the token-length boundary of the Gemma 4 E2B + LoRA illegal memory access.
# Results go to $HOME (not /tmp) because this host can be preempted.
set -u

PY=/usr/bin/python3.12
LORA=${LORA:-$HOME/model_ckpt/synthetic-loras/gemma-4-e2b-r16-stripped}
OUT=${OUT:-$HOME/e2b_ima_results}
mkdir -p "$OUT"

run() {
  local tag="$1" n="$2"
  local log="$OUT/$tag.log"
  timeout 1200 "$PY" scripts/repro_e2b_lora_ima.py \
    --max_model_len "$n" --lora "$LORA" > "$log" 2>&1
  local rc=$? v
  if grep -q 'ENGINE INIT OK' "$log"; then v=PASS
  elif grep -q 'illegal memory access' "$log"; then v='CRASH illegal-memory-access'
  elif grep -q 'TMA descriptor 700' "$log"; then v='CRASH tma-desc-700'
  elif grep -q 'OutOfMemoryError' "$log"; then v=OOM
  elif [ $rc -eq 124 ]; then v=TIMEOUT
  else v="OTHER rc=$rc"; fi
  printf '%-10s %-8s %s\n' "$tag" "$n" "$v" | tee -a "$OUT/summary.txt"
}

# Most informative points first: 2^31 / 24576 = 87381 is the predicted boundary.
for n in "${@:-87000 87700 80000}"; do
  run "p$n" "$n"
done
echo BISECT_DONE
