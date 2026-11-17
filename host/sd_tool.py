"""
microSD card operations on top_llm (see src/sources/sd_ctrl.sv).

  status                     STATUS, including the last SD operation
  load                       copy the card image into DDR
  store <addr> <nbytes>      copy DDR to the card (wipes whatever was on it)

usage: python3 host/sd_tool.py [--port /dev/ttyUSB0] status | load | store ADDR NBYTES
"""

import argparse as ap
import sys

from llm_link import Link

SD_ERRORS = {1: "CMD0 (no card / no response)", 2: "CMD8", 3: "ACMD41 (init timeout)",
             4: "CMD58", 5: "CMD16", 6: "read/write command rejected",
             7: "data token timeout", 8: "no valid image on the card",
             9: "checksum mismatch", 10: "write rejected", 11: "write busy timeout"}


def show(st: dict):
    sd = ("busy" if st["sd_busy"] else "ok" if st["sd_ok"] else
          f"error {st['sd_err_code']}: {SD_ERRORS.get(st['sd_err_code'], '?')}" if st["sd_err"]
          else "idle")
    print(f"DDR3 {'calibrated' if st['calibrated'] else 'NOT calibrated'}; "
          f"SD {sd}; " + (f"image {st['sd_bytes'] / 1e6:.2f} MB; " if st["sd_ok"] else "")
          + f"card detect pin {int(st['card_detect'])}")


def main(args) -> int:
    link = Link(args.port, args.baud)
    if args.cmd == "status":
        st = link.status()
    elif args.cmd == "load":
        st = link.sd_load()
    else:
        st = link.sd_store(int(args.addr, 0), int(args.nbytes, 0))
    show(st)
    return 0 if not st["sd_err"] else 1


if __name__ == "__main__":
    parser = ap.ArgumentParser(prog="sd_tool.py", description=__doc__.split("\n")[1])
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=3_000_000)
    parser.add_argument("cmd", choices=["status", "load", "store"])
    parser.add_argument("addr", nargs="?")
    parser.add_argument("nbytes", nargs="?")
    sys.exit(main(parser.parse_args()))
