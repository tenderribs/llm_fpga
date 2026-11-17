"""
Run Y = A @ B + C on the FPGA over the USB-UART and check it against numpy.

Protocol (see src/sources/top.sv): send A, B, C as N*N signed bytes each,
row-major; receive Y as N*N little-endian int32, row-major.

usage: python3 host/gemm_uart.py --port /dev/ttyUSB0 [--n 16] [--iters 10]
"""

import argparse as ap
import sys
import time

import numpy as np
import serial


def run_job(port: serial.Serial, A, B, C) -> np.ndarray:
    n = A.shape[0]
    payload = b"".join(m.astype(np.int8).tobytes() for m in (A, B, C))
    port.write(payload)
    out_bytes = 4 * n * n
    data = port.read(out_bytes)
    if len(data) != out_bytes:
        raise TimeoutError(f"got {len(data)} of {out_bytes} result bytes")
    return np.frombuffer(data, dtype="<i4").reshape(n, n)


def main(args) -> int:
    rng = np.random.default_rng(args.seed)
    n = args.n

    # Each job is 7*N*N bytes on the wire at 10 bits per byte; allow 2x that
    timeout = 2 * 7 * n * n * 10 / args.baud + 1.0

    with serial.Serial(args.port, args.baud, timeout=timeout) as port:
        # Let any partial job from an earlier run hit the FPGA's idle timeout,
        # then throw away anything still in the receive buffer
        time.sleep(0.2)
        port.reset_input_buffer()

        failures = 0
        for it in range(args.iters):
            if it == 0:
                # Identity first: Y must equal B, which is easy to eyeball
                A = np.identity(n, dtype=np.int32)
                C = np.zeros((n, n), dtype=np.int32)
            else:
                A = rng.integers(-128, 128, size=(n, n), dtype=np.int32)
                C = rng.integers(-128, 128, size=(n, n), dtype=np.int32)
            B = rng.integers(-128, 128, size=(n, n), dtype=np.int32)

            t0 = time.perf_counter()
            Y = run_job(port, A, B, C)
            dt = time.perf_counter() - t0
            Y_exp = A @ B + C

            bad = np.argwhere(Y != Y_exp)
            status = "PASS" if len(bad) == 0 else f"FAIL ({len(bad)} mismatches)"
            print(f"job {it}: {status}  {dt * 1e3:.1f} ms")
            for i, j in bad[:5]:
                print(f"  y[{i}][{j}]: got {Y[i, j]}, exp {Y_exp[i, j]}")
            failures += len(bad) != 0

    print("ALL PASSED" if failures == 0 else f"{failures} of {args.iters} jobs FAILED")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    parser = ap.ArgumentParser(prog="gemm_uart.py", description=__doc__.split("\n")[1])
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=115_200)
    parser.add_argument("--n", type=int, default=16, help="must match top.sv's N")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    sys.exit(main(parser.parse_args()))
