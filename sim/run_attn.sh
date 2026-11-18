#!/usr/bin/env bash
# attn.sv unit tests: generate vectors (sim/gen_attn.py), run tb_attn on each.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
VEC="$REPO/sim/build/vec/attn"
PYTHON=python3
[[ -x "$REPO/.venv/bin/python" ]] && PYTHON="$REPO/.venv/bin/python"
status=0
while read -r name mode hs nt nbeats nout; do
  echo "=== attn $name ==="
  "$REPO/sim/run_tb.sh" tb_attn "$name" -GVecDir="\"$VEC/$name\"" -GMode="$mode" \
    -GHS="$hs" -GNT="$nt" -GNBeats="$nbeats" -GNOut="$nout" || status=1
done < <("$PYTHON" "$REPO/sim/gen_attn.py" "$VEC")
exit $status
