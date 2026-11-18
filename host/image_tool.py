"""
Put a model image (host/build_image.py) into the board's DDR3 over the UART,
and optionally onto the microSD card so the FPGA loads it by itself at
power-on. Works with top_llm and top_soc (both have the UART command set).

  upload IMAGE [--store] [--verify]   write IMAGE to DDR offset 0
  check IMAGE                         compare DDR with IMAGE (header + samples)

usage: python3 host/image_tool.py [--port /dev/ttyUSB0] upload models/stories15M.img --store
"""

import argparse as ap
from pathlib import Path
import random
import sys
import time

from llm_link import Link

CHUNK = 1 << 16


def upload(link: Link, img: bytes):
    t0 = time.time()
    for off in range(0, len(img), CHUNK):
        link.write(off, img[off:off + CHUNK])
        if off % (16 * CHUNK) == 0:
            print(f"\ruploaded {(off + CHUNK) / 1e6:5.1f} / {len(img) / 1e6:.1f} MB", end="", flush=True)
    print(f"\ruploaded {len(img) / 1e6:.1f} MB in {time.time() - t0:.0f} s          ")


def check(link: Link, img: bytes, samples: int, full: bool) -> int:
    blocks = list(range(0, len(img), 4096))
    if not full:
        blocks = [0] + random.Random(1).sample(blocks[1:], min(samples, len(blocks) - 1))
    bad = 0
    for off in blocks:
        n = min(4096, len(img) - off)
        if link.read(off, n) != img[off:off + n]:
            bad += 1
            print(f"mismatch in the 4 KB block at 0x{off:08x}")
    print(f"checked {len(blocks)} blocks of 4 KB: {'ok' if not bad else f'{bad} differ'}")
    return bad


def main(args) -> int:
    img = Path(args.image).read_bytes()
    if img[:8] != b"LLMMODEL" or len(img) % 4096:
        sys.exit("not a model image")
    link = Link(args.port, args.baud)
    st = link.sd_wait()
    if not st["calibrated"]:
        sys.exit("DDR3 not calibrated")
    if args.cmd == "upload":
        upload(link, img)
    bad = check(link, img, 64, args.cmd == "upload" and args.verify)
    if bad:
        return 1
    if args.cmd == "upload" and args.store:
        t0 = time.time()
        st = link.sd_store(0, len(img))
        if st["sd_err"]:
            print(f"SD store failed, error {st['sd_err_code']} (see host/sd_tool.py)")
            return 1
        print(f"stored {len(img) / 1e6:.1f} MB on the microSD card in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    parser = ap.ArgumentParser(prog="image_tool.py", description=__doc__.split("\n")[1])
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=3_000_000)
    parser.add_argument("cmd", choices=["upload", "check"])
    parser.add_argument("image")
    parser.add_argument("--store", action="store_true", help="then store it on the microSD card")
    parser.add_argument("--verify", action="store_true", help="read all of it back")
    sys.exit(main(parser.parse_args()))
