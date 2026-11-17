#!/usr/bin/env bash
# Build and run a testbench with Verilator (5.x, needs --timing for the TB's
# delays and @(posedge) waits).
# Usage: [TB=tb_top] [VERBOSE=1] sim/run_verilator.sh [N ...]   (default: 2)
#   TB  testbench in src/sim/ (default tb_systolic_array)
# Logs per run: sim/build/verilator/<TB>/N<N>_<mode>/{build,sim}.log
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
TB=${TB:-tb_systolic_array}
BUILD="$REPO/sim/build/verilator/$TB"

# Use the repo venv if there is one
PYTHON=python3
[[ -x "$REPO/.venv/bin/python" ]] && PYTHON="$REPO/.venv/bin/python"

# top_llm.sv is left out: it needs the DDR controller IP (simulate
# llm_core instead, see tb_llm_core)
SOURCES=("$REPO/sim/IBUFDS.v" "$REPO/sim/BUFG.v")
while IFS= read -r f; do SOURCES+=("$f"); done \
  < <(find "$REPO/src/sources" -name '*.sv' ! -name 'top_llm.sv' | sort)
SOURCES+=("$REPO/src/sim/$TB.sv")

if (($#)); then SIZES=("$@"); else SIZES=(2); fi
MODES=(iden rand)

"$PYTHON" "$REPO/sim/gen_stim.py" --sizes "${SIZES[@]}"

# Print the first line of every Verilator diagnostic in a log (one line per
# warning/error, with file:line), repo paths shortened. Full context stays in
# the log.
show_diags() {
  grep -E '^%(Warning|Error)' "$1" | sed "s|$REPO/||g" || true
}

status=0
for N in "${SIZES[@]}"; do
  for MODE in "${MODES[@]}"; do
    echo "=== N=$N mode=$MODE ==="
    VEC="$REPO/sim/N$N"
    OUT="$BUILD/N${N}_$MODE"
    rm -rf "$OUT"
    mkdir -p "$OUT"

    verilator --binary --timing -Wno-fatal -Wall -Wno-TIMESCALEMOD \
      --trace-fst -j "$(nproc)" \
      --x-assign unique --x-initial unique \
      -Mdir "$OUT" \
      -GN="$N" \
      -GAFile="\"$VEC/A_${MODE}_$N.hex\"" \
      -GBFile="\"$VEC/B_${MODE}_$N.hex\"" \
      -GCFile="\"$VEC/C_${MODE}_$N.hex\"" \
      -GYFile="\"$VEC/Y_${MODE}_$N.hex\"" \
      -GFstPath="\"$OUT/out.fst\"" \
      --top-module "$TB" "${SOURCES[@]}" > "$OUT/build.log" 2>&1 \
      || {
        show_diags "$OUT/build.log"
        echo "BUILD FAILED (full log: ${OUT#"$REPO"/}/build.log)"
        status=1
        continue
      }
    show_diags "$OUT/build.log"

    # Full sim output (including $display debug prints) goes to sim.log;
    # set VERBOSE=1 to also echo all of it here.
    "$OUT/V$TB" +verilator+seed+$RANDOM +verilator+rand+reset+2 \
      > "$OUT/sim.log" 2>&1 || true
    if [[ ${VERBOSE:-0} == 1 ]]; then
      cat "$OUT/sim.log"
    else
      grep -E '^(TEST|MISMATCH|FAIL|TIMEOUT)|%Error' "$OUT/sim.log" || true
    fi
    grep -q 'TEST PASSED' "$OUT/sim.log" || status=1
  done
done

exit $status
