"""
Run real llama2.c matmuls on the FPGA (top_qmv) over the USB-UART and check
every output bit-exact against runq.c.

Inputs are the matmul dumps from ref/build_ref.sh (ref/build/stories15M/
mm_NNN.bin); see src/sources/top_qmv.sv for the wire protocol.

usage: python3 host/qmv_uart.py [--port /dev/ttyUSB0] [dump index ...]
       (default dumps: 000 004 006 = layer 0 wq, w1, w2)
"""

import argparse as ap
from pathlib import Path
import struct
import sys
import time

import numpy as np
import serial

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "ref"))
import q80  # noqa: E402

GS_HW = 32  # top_qmv build parameters
P = 32
R = 16


def pack_job(x: q80.QTensor, w: q80.QTensor, n: int, d: int, gs: int) -> bytes:
    wq, ws = q80.qmv_stream(w, n, d, gs, R, P)
    beats = np.empty((len(wq), P + 4), np.uint8)
    beats[:, :P] = wq.view(np.uint8)
    beats[:, P:] = ws.astype("<f4").view(np.uint8).reshape(-1, 4)
    return b"".join([
        struct.pack("<HH", n, d),
        x.q.tobytes(),
        x.s.astype("<f4").tobytes(),
        beats.tobytes(),
    ])


def run_job(port: serial.Serial, payload: bytes, d: int) -> np.ndarray:
    port.write(payload)
    data = port.read(4 * d)
    if len(data) != 4 * d:
        raise TimeoutError(f"got {len(data)} of {4 * d} result bytes")
    return np.frombuffer(data, "<f4")


def main(args) -> int:
    failures = 0
    with serial.Serial(args.port, args.baud, timeout=5.0) as port:
        time.sleep(0.2)  # let any half-sent job from an earlier run time out
        port.reset_input_buffer()

        for idx in args.dumps:
            path = REPO / f"ref/build/stories15M/mm_{idx}.bin"
            x, w, y_ref, n, d, gs = q80.read_matmul(str(path))
            if gs != GS_HW:
                print(f"mm_{idx}: group size {gs}, but the bitstream is built for {GS_HW}")
                failures += 1
                continue

            payload = pack_job(x, w, n, d, gs)
            # the port timeout only covers gaps; results arrive while sending
            t0 = time.perf_counter()
            y = run_job(port, payload, d)
            dt = time.perf_counter() - t0

            bad = np.flatnonzero(y.view(np.uint32) != y_ref.view(np.uint32))
            status = "PASS (bit-exact)" if len(bad) == 0 else f"FAIL ({len(bad)} of {d} differ)"
            print(f"mm_{idx} {d}x{n}: {status}  {len(payload) / 1e3:.0f} kB in {dt:.2f} s "
                  f"({len(payload) * 10 / dt / 1e6:.2f} Mbit/s on the wire)")
            for i in bad[:5]:
                print(f"  y[{i}]: got {y[i]!r}, exp {y_ref[i]!r}")
            failures += len(bad) != 0

    print("ALL PASSED" if failures == 0 else f"{failures} of {len(args.dumps)} FAILED")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    parser = ap.ArgumentParser(prog="qmv_uart.py", description=__doc__.split("\n")[1])
    parser.add_argument("dumps", nargs="*", default=["000", "004", "006"])
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=1_000_000, help="must match top_qmv's Baud")
    sys.exit(main(parser.parse_args()))
