"""
Software stand-in for top_llm on a pseudo-terminal, to test host programs
(host/llm_host.c, host/llm_link.py) without the board.

Implements llm_core.sv's commands (STATUS, WRITE, READ, MATMUL, SD_LOAD,
SD_STORE; the "card" lives in memory and SD operations finish at once). MATMUL
decodes the qmv DDR format back into W the way wstream_reader + qmv consume
it, and computes y with q80.matmul (bit-exact with runq.c). So a host that
packs or addresses weights wrongly gets wrong results here too.

usage: python3 sim/fake_board.py      prints the pty path to connect to,
                                      serves until killed
"""

import os
from pathlib import Path
import struct
import sys
import tty

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ref"))
import q80  # noqa: E402

GS, P, R, BEAT = 32, 32, 16, 32


class Board:
    def __init__(self, fd):
        self.fd = fd
        self.ddr = {}  # 4 KB page number -> bytearray
        self.card = None  # (ddr_addr, image bytes) stored on the "SD card"
        self.sd_flags, self.sd_code, self.sd_bytes = 0, 0, 0

    def read(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = os.read(self.fd, n - len(buf))
            if not chunk:
                raise EOFError
            buf += chunk
        return buf

    def write(self, data: bytes):
        while data:
            k = os.write(self.fd, data)
            data = data[k:]

    def ddr_write(self, addr: int, data: bytes):
        for off in range(0, len(data), 4096):
            a = addr + off
            page = self.ddr.setdefault(a >> 12, bytearray(4096))
            k = min(4096 - (a & 0xFFF), len(data) - off)
            page[a & 0xFFF:(a & 0xFFF) + k] = data[off:off + k]
            if k < min(4096, len(data) - off):  # crossed into the next page
                rest = data[off + k:off + 4096]
                nxt = self.ddr.setdefault((a >> 12) + 1, bytearray(4096))
                nxt[:len(rest)] = rest

    def ddr_read(self, addr: int, n: int) -> bytes:
        out = bytearray()
        while len(out) < n:
            a = addr + len(out)
            page = self.ddr.get(a >> 12, bytes(4096))
            k = min(4096 - (a & 0xFFF), n - len(out))
            out += page[a & 0xFFF:(a & 0xFFF) + k]
        return bytes(out)

    def matmul(self, addr: int, n: int, d: int, xq: bytes, xs: bytes) -> bytes:
        beats = d * n // P
        raw = np.frombuffer(self.ddr_read(addr, beats // 8 * 9 * BEAT), np.uint8)
        chunks = raw.reshape(beats // 8, 9, BEAT)
        ws = chunks[:, 0, :].copy().view("<f4").reshape(-1)  # one per beat
        wq = chunks[:, 1:, :].reshape(beats, P).view(np.int8)
        # undo qmv_stream: (block, group, row, beat-in-group) -> row-major
        G, bpg = n // GS, GS // P
        q = wq.reshape(d // R, G, R, bpg, P).transpose(0, 2, 1, 3, 4).reshape(d, n)
        s = ws.reshape(d // R, G, R, bpg)[..., 0].transpose(0, 2, 1).reshape(d * G)
        x = q80.QTensor(np.frombuffer(xq, np.int8), np.frombuffer(xs, "<f4"))
        w = q80.QTensor(q.reshape(-1), s)
        return q80.matmul(x, w, n, d, GS).astype("<f4").tobytes()

    def serve(self):
        while True:
            op = self.read(1)[0]
            if op == 0x00:
                self.write(b"LLM" + bytes([0x01 | self.sd_flags, self.sd_code, 0, 0, 0])
                           + struct.pack("<I", self.sd_bytes))
            elif op == 0x01:
                addr, nb = struct.unpack("<II", self.read(8))
                self.ddr_write(addr, self.read(nb * BEAT))
                self.write(b"\x01")
            elif op == 0x02:
                addr, nb = struct.unpack("<II", self.read(8))
                self.write(self.ddr_read(addr, nb * BEAT))
            elif op == 0x03:
                addr, n, d = struct.unpack("<IHH", self.read(8))
                xq = self.read(n)
                xs = self.read(n // GS * 4)
                self.write(self.matmul(addr, n, d, xq, xs))
            elif op == 0x04:  # SD_LOAD (instant here)
                self.write(b"\x04")
                if self.card is None:
                    self.sd_flags, self.sd_code = 0x08, 8  # bad header
                else:
                    addr, img = self.card
                    self.ddr_write(addr, img)
                    self.sd_flags, self.sd_code, self.sd_bytes = 0x04, 0, len(img)
            elif op == 0x05:  # SD_STORE
                addr, nbytes = struct.unpack("<II", self.read(8))
                self.write(b"\x05")
                self.card = (addr, self.ddr_read(addr, nbytes))
                self.sd_flags, self.sd_code, self.sd_bytes = 0x04, 0, nbytes
            # unknown opcodes are ignored, as on the FPGA


if __name__ == "__main__":
    master, slave = os.openpty()
    tty.setraw(master)
    print(os.ttyname(slave), flush=True)
    try:
        Board(master).serve()
    except (EOFError, OSError):
        pass
