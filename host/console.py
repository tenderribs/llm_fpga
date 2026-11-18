"""
Print the FPGA firmware's console (UART) until a line matching --until
appears, or until --timeout seconds pass without output.

usage: python3 host/console.py [--port /dev/ttyUSB0] [--until 'PASS|FAIL'] [--timeout 30]
Exit status 0 if the --until line was seen and doesn't contain FAIL.
"""

import argparse as ap
import re
import sys
import time

import serial


def main(args) -> int:
    pat = re.compile(args.until) if args.until else None
    with serial.Serial(args.port, args.baud, timeout=0.2) as port:
        line = b""
        last = time.time()
        while time.time() - last < args.timeout:
            data = port.read(4096)
            if not data:
                continue
            last = time.time()
            sys.stdout.write(data.decode(errors="replace"))
            sys.stdout.flush()
            line += data
            *done, line = line.split(b"\n")
            for l in done:
                if pat and pat.search(l.decode(errors="replace")):
                    return 1 if b"FAIL" in l else 0
    print(f"\n[console: no output for {args.timeout} s]")
    return 2


if __name__ == "__main__":
    parser = ap.ArgumentParser(prog="console.py", description=__doc__.split("\n")[1])
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=3_000_000)
    parser.add_argument("--until", default=r"PASS|FAIL")
    parser.add_argument("--timeout", type=float, default=30)
    sys.exit(main(parser.parse_args()))
