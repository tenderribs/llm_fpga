#!/usr/bin/env bash
# Build and run one testbench with Verilator.
# Usage: sim/run_tb.sh <tb_name> <run_tag> [verilator -G args ...]
#   builds src/sim/<tb_name>.sv with every file in src/sources/ and the
#   simulation models in src/sim/ (files not named tb_*) into
#   sim/build/verilator/<tb_name>/<run_tag>/, runs it, and prints the
#   TEST / MISMATCH / FAIL / TIMEOUT lines. Exit status 0 only on TEST PASSED.
#   VERBOSE=1 prints the whole sim log; TRACE=1 writes a waveform (out.fst).
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
TB=$1
TAG=$2
shift 2
OUT="$REPO/sim/build/verilator/$TB/$TAG"

# top_llm.sv is left out: it needs the DDR controller IP (simulate
# llm_core instead, see tb_llm_core)
SOURCES=("$REPO/sim/IBUFDS.v" "$REPO/sim/BUFG.v")
while IFS= read -r f; do SOURCES+=("$f"); done \
  < <(find "$REPO/src/sources" -name '*.sv' ! -name 'top_llm.sv' | sort)
# simulation models (src/sim/ files that aren't testbenches), then the TB
while IFS= read -r f; do SOURCES+=("$f"); done \
  < <(find "$REPO/src/sim" -name '*.sv' ! -name 'tb_*' | sort)
SOURCES+=("$REPO/src/sim/$TB.sv")

TRACE_ARGS=()
[[ ${TRACE:-0} == 1 ]] && TRACE_ARGS=(--trace-fst)

rm -rf "$OUT"
mkdir -p "$OUT"
if ! verilator --binary --timing -Wno-fatal -Wno-TIMESCALEMOD -O3 -j "$(nproc)" \
    "${TRACE_ARGS[@]}" --x-assign unique --x-initial unique \
    -Mdir "$OUT" --top-module "$TB" "$@" "${SOURCES[@]}" > "$OUT/build.log" 2>&1; then
  grep -E '^%Error' "$OUT/build.log" | sed "s|$REPO/||g" || true
  echo "BUILD FAILED (full log: ${OUT#"$REPO"/}/build.log)"
  exit 1
fi

"$OUT/V$TB" +verilator+seed+$RANDOM +verilator+rand+reset+2 > "$OUT/sim.log" 2>&1 || true
if [[ ${VERBOSE:-0} == 1 ]]; then
  cat "$OUT/sim.log"
else
  grep -E '^(TEST|MISMATCH|FAIL|TIMEOUT|checked)|%Error' "$OUT/sim.log" || true
fi
grep -q 'TEST PASSED' "$OUT/sim.log"
