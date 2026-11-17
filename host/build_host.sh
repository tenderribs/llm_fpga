#!/usr/bin/env bash
# Build host/llm_host (runq.c with matmul on the FPGA) into ref/build/.
# Needs ref/build/runq_patched.c from ref/build_ref.sh.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
BUILD="$REPO/ref/build"
[[ -f "$BUILD/runq_patched.c" ]] || { echo "run ref/build_ref.sh first"; exit 1; }
# Same flags as runq_dump: no -ffast-math, no OpenMP, so the CPU-side float
# math matches the reference run exactly.
gcc -O2 -Wall -Wno-unused-result -I"$BUILD" -o "$BUILD/llm_host" "$REPO/host/llm_host.c" -lm
echo "built ${BUILD#"$REPO"/}/llm_host"
