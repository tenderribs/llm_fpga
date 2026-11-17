#!/usr/bin/env bash
# Build the golden reference and generate its outputs into ref/build/.
#
#   1. quantize models/stories15M.bin to models/stories15M_q80.bin (Q8_0)
#   2. build runq_dump: llama2.c's runq.c with matmul() hooked
#   3. run it greedily, dumping logits and every matmul at one position
#
# Download the model first:
#   mkdir -p models && curl -L -o models/stories15M.bin \
#     https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
#
# Usage: ref/build_ref.sh [prompt] [steps] [dump_pos]
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
LLAMA="$REPO/third_party/llama2.c"
BUILD="$REPO/ref/build"
OUT="$BUILD/stories15M"
PROMPT=${1:-"Once upon a time"}
STEPS=${2:-64}
DUMP_POS=${3:-0}

PYTHON=python3
[[ -x "$REPO/.venv/bin/python" ]] && PYTHON="$REPO/.venv/bin/python"

mkdir -p "$BUILD" "$OUT"

if [[ ! -f "$REPO/models/stories15M_q80.bin" ]]; then
  "$PYTHON" "$REPO/ref/export_q80.py" "$REPO/models/stories15M.bin" \
    "$REPO/models/stories15M_q80.bin"
fi

# Rename runq.c's matmul so runq_dump.c can wrap it, and declare the wrapper
# before forward() calls it. No -ffast-math and no OpenMP: the float math must
# run in source order for the dumps to be a bit-exact golden.
sed -e 's/^void matmul(/void matmul_orig(/' \
    -e 's/^float\* forward(/void matmul(float*, QuantizedTensor*, QuantizedTensor*, int, int);\n&/' \
    "$LLAMA/runq.c" > "$BUILD/runq_patched.c"
gcc -O2 -Wall -Wno-unused-result -I"$BUILD" -o "$BUILD/runq_dump" "$REPO/ref/runq_dump.c" -lm

rm -f "$OUT"/mm_*.bin
"$BUILD/runq_dump" "$REPO/models/stories15M_q80.bin" "$LLAMA/tokenizer.bin" \
  "$PROMPT" "$STEPS" "$DUMP_POS" "$OUT"
echo "wrote $(ls "$OUT"/mm_*.bin | wc -l) matmul dumps, logits and tokens to ${OUT#"$REPO"/}"
