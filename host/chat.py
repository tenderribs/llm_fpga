"""
Talk to the llama2 firmware on top_soc (fw/llm) over the UART: send a
prompt, print the story as the board generates it.

  python3 host/chat.py                       interactive: one prompt per line
  python3 host/chat.py "Once upon a time"    one prompt, then exit
  python3 host/chat.py --steps 64 "A cat"    at most 64 tokens
  python3 host/chat.py --trace "Once upon a time" > run.txt
                                             full-length run plus the trace
                                             lines (compare with fw/build/golden)

The prompt goes to the board with llm_core's CONSOLE_IN command; the board's
replies are its console output, up to "LLM DONE".
"""

import argparse as ap
import sys
import time

import serial

OP_CONSOLE_IN = 0x06


class Board:
    def __init__(self, port: str, baud: int):
        self.ser = serial.Serial(port, baud, timeout=0.2)
        self.buf = b""

    def send_line(self, text: str):
        data = (text + "\n").encode()
        for i in range(0, len(data), 255):
            chunk = data[i:i + 255]
            self.ser.write(bytes([OP_CONSOLE_IN, len(chunk)]) + chunk)

    def lines_until(self, marker: bytes, timeout: float, echo):
        """Read console output until a line equal to marker; echo(text)
        gets everything before it as it arrives."""
        last = time.time()
        while True:
            data = self.ser.read(4096)
            if data:
                last = time.time()
                self.buf += data
            # pass on all but a possibly incomplete marker line
            while True:
                nl = self.buf.find(b"\n")
                if nl < 0:
                    if self.buf and not marker.startswith(self.buf):
                        echo(self.buf.decode(errors="replace"))
                        self.buf = b""
                    break
                line = self.buf[:nl + 1]
                self.buf = self.buf[nl + 1:]
                if line.rstrip(b"\r\n") == marker:
                    return True
                echo(line.decode(errors="replace"))
            if time.time() - last > timeout:
                return False


def main(args) -> int:
    board = Board(args.port, args.baud)
    out = lambda s: (sys.stdout.write(s), sys.stdout.flush())
    quiet = lambda s: None

    def run(line: str) -> bool:
        board.send_line(line)
        ok = board.lines_until(b"LLM DONE", args.timeout, out)
        board.lines_until(b"READY", args.timeout, quiet)
        if not ok:
            print("\n[no answer from the board: is top_soc running fw/llm?]")
        return ok

    # the firmware says READY when it waits for a line; it may already have
    # said so (nobody listening), so don't wait for it
    if args.steps:
        if not run(f"/steps {args.steps}"):
            return 1
    if args.prompt is not None:
        return 0 if run(("/trace " if args.trace else "") + args.prompt) else 1
    print("prompt (empty line or Ctrl-D to quit)")
    while True:
        try:
            line = input("> ")
        except EOFError:
            print()
            return 0
        if not line:
            return 0
        if not run(line):
            return 1
        print()


if __name__ == "__main__":
    parser = ap.ArgumentParser(prog="chat.py", description=__doc__.split("\n")[1])
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=3_000_000)
    parser.add_argument("--steps", type=int, help="most tokens per story")
    parser.add_argument("--trace", action="store_true", help="full length, with trace lines")
    parser.add_argument("--timeout", type=float, default=30, help="seconds without output")
    parser.add_argument("prompt", nargs="?")
    sys.exit(main(parser.parse_args()))
