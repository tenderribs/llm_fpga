"""
Bring-up test for top_llm on the board:
  1. STATUS: the DDR3 calibrated
  2. memory test: random data written and read back at a few addresses
     spread over the 1 GB
  3. matmuls from DDR: real runq.c matmuls (ref/build_ref.sh dumps), weights
     written to DDR in qmv DDR format, every result checked bit-exact

usage: python3 host/ddr_test.py [--port /dev/ttyUSB0] [dump index ...]
       (default dumps: 000 004 006)
"""

import argparse as ap
from pathlib import Path
import sys
import time

import numpy as np

from llm_link import Link, ddr_format, GS_HW

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "ref"))
import q80  # noqa: E402


def main(args) -> int:
    ok = True
    link = Link(args.port, args.baud)

    st = link.sd_wait()  # a boot-time SD load may still be running
    print(f"status: DDR3 {'calibrated' if st['calibrated'] else 'NOT calibrated'}")
    if not st["calibrated"]:
        return 1

    rng = np.random.default_rng(0)
    for addr in [0x0, 0x1234_5600, 0x2000_0000, 0x3FFF_F000]:
        data = rng.integers(0, 256, 4096, dtype=np.uint8).tobytes()
        t0 = time.perf_counter()
        link.write(addr, data)
        back = link.read(addr, len(data))
        dt = time.perf_counter() - t0
        good = back == data
        ok &= good
        print(f"memtest 0x{addr:08X}: {'PASS' if good else 'FAIL'} (4 KB written + read in {dt:.2f} s)")

    addr = 0x100_0000
    for idx in args.dumps:
        x, w, y_ref, n, d, gs = q80.read_matmul(str(REPO / f"ref/build/stories15M/mm_{idx}.bin"))
        if gs != GS_HW:
            print(f"mm_{idx}: group size {gs}, the bitstream is built for {GS_HW}")
            ok = False
            continue
        blob = ddr_format(w, n, d, gs)
        t0 = time.perf_counter()
        link.write(addr, blob)
        t_load = time.perf_counter() - t0
        t0 = time.perf_counter()
        y = link.matmul(addr, x, n, d)
        t_mm = time.perf_counter() - t0
        bad = np.flatnonzero(y.view(np.uint32) != y_ref.view(np.uint32))
        good = len(bad) == 0
        ok &= good
        print(f"mm_{idx} {d}x{n} from DDR: {'PASS (bit-exact)' if good else f'FAIL ({len(bad)} differ)'}"
              f"  load {len(blob) / 1e3:.0f} kB in {t_load:.2f} s, matmul round trip {t_mm * 1e3:.1f} ms")
        for i in bad[:5]:
            print(f"  y[{i}]: got {y[i]!r}, exp {y_ref[i]!r}")
        addr += (len(blob) + 4095) // 4096 * 4096

    link.close()
    print("ALL PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    parser = ap.ArgumentParser(prog="ddr_test.py", description=__doc__.split("\n")[1])
    parser.add_argument("dumps", nargs="*", default=["000", "004", "006"])
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=3_000_000)
    sys.exit(main(parser.parse_args()))
