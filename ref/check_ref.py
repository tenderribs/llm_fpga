"""
Check the numpy reference (ref/q80.py) against the runq.c golden dumps that
ref/build_ref.sh writes:

  1. matmul: every dumped call must match bit-for-bit
  2. forward: run the same tokens through Model.forward and compare.
     This can't be bit-exact: numpy's float sums and exp/sqrt differ from C's
     by an ulp now and then, and once such an ulp tips an activation across
     a quantization rounding boundary, that int8 differs by 1 and the
     difference stays in the KV cache. So: logits within 1.0, and the same
     greedy token wherever runq.c's top two logits are more than 0.1 apart.

usage: python3 ref/check_ref.py [ref/build/stories15M]
"""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import q80  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def main(out_dir: Path) -> int:
    ok = True

    # 1. matmul, bit-exact
    dumps = sorted(out_dir.glob("mm_*.bin"))
    bad = 0
    for p in dumps:
        x, w, out, n, d, gs = q80.read_matmul(str(p))
        got = q80.matmul(x, w, n, d, gs)
        if not np.array_equal(got.view(np.uint32), out.view(np.uint32)):
            diff = np.count_nonzero(got != out)
            print(f"MISMATCH {p.name} ({d}x{n}): {diff} of {d} outputs differ")
            bad += 1
    print(f"matmul: {len(dumps) - bad}/{len(dumps)} dumps bit-exact")
    ok &= bad == 0 and len(dumps) > 0

    # 2. full forward
    model = q80.Model(str(REPO / "models/stories15M_q80.bin"))
    tokens = [int(t) for t in (out_dir / "tokens.txt").read_text().split()]
    logits_c = np.fromfile(out_dir / "logits.bin", "<f4").reshape(-1, model.vocab_size)
    steps = logits_c.shape[0]

    max_err = 0.0
    first_jump = None  # first step where the error leaves float-noise level
    near_ties = 0
    real_diverge = []
    for pos in range(steps):
        logits = model.forward(tokens[pos], pos)
        err = float(np.abs(logits - logits_c[pos]).max())
        max_err = max(max_err, err)
        if first_jump is None and err > 1e-4:
            first_jump = pos
        if logits.argmax() != logits_c[pos].argmax():
            top2 = np.sort(logits_c[pos])[-2:]
            if top2[1] - top2[0] < 0.1:
                near_ties += 1
            else:
                real_diverge.append(pos)
    print(f"forward: {steps} steps, max |logit diff| = {max_err:.2e} "
          f"(float noise until step {first_jump if first_jump is not None else steps}), "
          f"{near_ties} near-tie token flips, {len(real_diverge)} real divergences {real_diverge}")
    ok &= max_err < 1.0 and not real_diverge

    print("PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "ref/build/stories15M"))
