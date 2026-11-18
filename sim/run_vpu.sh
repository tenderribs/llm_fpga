#!/usr/bin/env bash
# vpu.sv unit tests: generate vectors (sim/gen_vpu.py), run tb_vpu on each.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
VEC="$REPO/sim/build/vec/vpu"
PYTHON=python3
[[ -x "$REPO/.venv/bin/python" ]] && PYTHON="$REPO/.venv/bin/python"
status=0
while read -r name op n nin nout fparam; do
  echo "=== vpu $name ==="
  "$REPO/sim/run_tb.sh" tb_vpu "$name" -GVecDir="\"$VEC/$name\"" -GOp="$op" -GN="$n" \
    -GNIn="$nin" -GNOut="$nout" -GFParam="32'h$fparam" || status=1
done < <("$PYTHON" "$REPO/sim/gen_vpu.py" "$VEC")
exit $status
