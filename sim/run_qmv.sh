#!/usr/bin/env bash
# Tests for the quantized matmul engine:
#   1. tb_fp32: float units against numpy float32
#   2. tb_qmv:  qmv on real runq.c matmuls, bit-exact
#   3. tb_top_qmv: top_qmv through its UART pins, on one matmul
#   4. tb_llm_core: llm_core's commands, matmuls with weights read from an
#      AXI memory model (needs dumps 000 and 006)
# Needs the golden dumps from ref/build_ref.sh.
# Usage: sim/run_qmv.sh [dump index ...]   (default: 000 004 006 042)
#   plus a random 256x512 matmul with group size 64, always
#   000 = layer 0 wq (288x288), 004 = w1 (768x288), 006 = w2 (288x768),
#   042 = classifier (32000x288)
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
VEC="$REPO/sim/build/vec"
DUMPS="$REPO/ref/build/stories15M"
PYTHON=python3
[[ -x "$REPO/.venv/bin/python" ]] && PYTHON="$REPO/.venv/bin/python"

if (($#)); then IDX=("$@"); else IDX=(000 004 006 042); fi
status=0

echo "=== fp32 units ==="
"$PYTHON" "$REPO/sim/gen_qmv.py" fp "$VEC/fp" > /dev/null
"$REPO/sim/run_tb.sh" tb_fp32 default -GVecDir="\"$VEC/fp\"" \
  -GNumAdd="$(wc -l < "$VEC/fp/add.hex")" -GNumMul="$(wc -l < "$VEC/fp/mul.hex")" \
  -GNumI2f="$(wc -l < "$VEC/fp/i2f.hex")" -GNumDiv="$(wc -l < "$VEC/fp/div.hex")" \
  -GNumExp="$(wc -l < "$VEC/fp/exp.hex")" || status=1

# group size 64 (two beats per group), which stories15M doesn't use
"$PYTHON" "$REPO/sim/gen_qmv.py" rand-dump "$DUMPS/mm_rand_gs64.bin" 512 256 64
IDX+=(rand_gs64)

for i in "${IDX[@]}"; do
  out="$VEC/mm_$i"
  # prints "n=... d=... gs=..., ..."; pull the three numbers out
  read -r n d gs < <("$PYTHON" "$REPO/sim/gen_qmv.py" mm "$out" "$DUMPS/mm_$i.bin" |
    sed -E 's/.*n=([0-9]+) d=([0-9]+) gs=([0-9]+).*/\1 \2 \3/')
  echo "=== qmv mm_$i: ${d}x${n}, group size $gs ==="
  "$REPO/sim/run_tb.sh" tb_qmv "mm_$i" -GVecDir="\"$out\"" -GN="$n" -GD="$d" -GGS="$gs" \
    || status=1
done

# whole design through the UART pins, on the smallest dump
out="$VEC/mm_000"
echo "=== top_qmv over UART: mm_000 ==="
"$REPO/sim/run_tb.sh" tb_top_qmv mm_000 -GVecDir="\"$out\"" -GN=288 -GD=288 -GGS=32 \
  || status=1

# llm_core (UART commands, DDR weight reader, qmv) with an AXI memory model
"$PYTHON" "$REPO/sim/gen_attn.py" "$VEC/attn" > /dev/null
echo "=== llm_core: STATUS, WRITE/READ, matmuls from DDR ==="
"$REPO/sim/run_tb.sh" tb_llm_core default \
  -GVecDir0="\"$VEC/mm_000\"" -GN0=288 -GD0=288 \
  -GVecDir1="\"$VEC/mm_006\"" -GN1=768 -GD1=288 \
  -GVecDirAttS="\"$VEC/attn/s48_37\"" -GVecDirAttW="\"$VEC/attn/w48_37\"" || status=1

exit $status
