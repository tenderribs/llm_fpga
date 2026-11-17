"""
Host side of top_llm's UART command interface (see src/sources/llm_core.sv),
and the "qmv DDR format" weights are stored in.
"""

from pathlib import Path
import struct
import sys
import time

import numpy as np
import serial

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ref"))
import q80  # noqa: E402

OP_STATUS, OP_WRITE, OP_READ, OP_MATMUL, OP_SD_LOAD, OP_SD_STORE = 0, 1, 2, 3, 4, 5
BEAT = 32  # bytes per DDR beat
GS_HW = 32  # top_llm build parameters
P = 32
R = 16


def ddr_format(w: q80.QTensor, n: int, d: int, gs: int) -> bytes:
    """W in qmv DDR format (see wstream_reader.sv): the qmv beat stream in
    chunks of 8 beats, each one beat of the 8 scales then the 8 weight beats."""
    wq, ws = q80.qmv_stream(w, n, d, gs, R, P)
    nb = len(wq)
    assert nb % 8 == 0
    chunks = np.empty((nb // 8, 9, BEAT), np.uint8)
    chunks[:, 0, :] = ws.astype("<f4").view(np.uint8).reshape(nb // 8, BEAT)
    chunks[:, 1:, :] = wq.view(np.uint8).reshape(nb // 8, 8, BEAT)
    return chunks.tobytes()


class Link:
    def __init__(self, port: str = "/dev/ttyUSB0", baud: int = 3_000_000, timeout: float = 5.0):
        self.ser = serial.Serial(port, baud, timeout=timeout)
        time.sleep(0.2)  # let any half-sent command from an earlier run time out
        self.ser.reset_input_buffer()

    def close(self):
        self.ser.close()

    def _read(self, n: int) -> bytes:
        data = self.ser.read(n)
        if len(data) != n:
            raise TimeoutError(f"got {len(data)} of {n} reply bytes")
        return data

    def status(self) -> dict:
        """calibrated: DDR3 ready; sd_busy / sd_ok / sd_err: the last SD
        operation (sd_err_code, see sd_ctrl.sv); card_detect: the slot's CD
        pin; sd_bytes: size of the last image loaded or stored."""
        self.ser.write(bytes([OP_STATUS]))
        r = self._read(12)
        if r[:3] != b"LLM":
            raise RuntimeError(f"bad status reply {r!r}: is top_llm loaded?")
        f = r[3]
        return dict(calibrated=bool(f & 1), sd_busy=bool(f & 2), sd_ok=bool(f & 4),
                    sd_err=bool(f & 8), card_detect=bool(f & 16), sd_err_code=r[4],
                    sd_bytes=struct.unpack("<I", r[8:12])[0])

    def sd_wait(self, timeout: float = 600.0) -> dict:
        """Poll STATUS until no SD operation is running."""
        t0 = time.time()
        while True:
            st = self.status()
            if not st["sd_busy"]:
                return st
            if time.time() - t0 > timeout:
                raise TimeoutError("SD operation still running")
            time.sleep(0.2)

    def sd_load(self) -> dict:
        """Copy the card image into DDR; returns the final status."""
        self.ser.write(bytes([OP_SD_LOAD]))
        if self._read(1)[0] != OP_SD_LOAD:
            raise RuntimeError("bad SD_LOAD ack (an SD operation already running?)")
        return self.sd_wait()

    def sd_store(self, addr: int, nbytes: int) -> dict:
        """Copy nbytes of DDR at addr to the card; returns the final status."""
        self.ser.write(struct.pack("<BII", OP_SD_STORE, addr, nbytes))
        if self._read(1)[0] != OP_SD_STORE:
            raise RuntimeError("bad SD_STORE ack (an SD operation already running?)")
        return self.sd_wait()

    def write(self, addr: int, data: bytes):
        assert addr % BEAT == 0 and len(data) % BEAT == 0
        self.ser.write(struct.pack("<BII", OP_WRITE, addr, len(data) // BEAT) + data)
        # the ack comes once everything is written; allow for the transfer
        old = self.ser.timeout
        self.ser.timeout = len(data) * 10 / self.ser.baudrate * 1.5 + 2
        r = self._read(1)
        self.ser.timeout = old
        if r[0] != OP_WRITE:
            raise RuntimeError(f"bad write ack {r!r}")

    def read(self, addr: int, nbytes: int) -> bytes:
        assert addr % BEAT == 0 and nbytes % BEAT == 0
        self.ser.write(struct.pack("<BII", OP_READ, addr, nbytes // BEAT))
        return self._read(nbytes)

    def matmul(self, addr: int, x: q80.QTensor, n: int, d: int) -> np.ndarray:
        """y = W @ x with W (d x n) stored at addr in qmv DDR format."""
        assert addr % 4096 == 0
        self.ser.write(struct.pack("<BIHH", OP_MATMUL, addr, n, d)
                       + x.q.astype(np.int8).tobytes() + x.s.astype("<f4").tobytes())
        return np.frombuffer(self._read(4 * d), "<f4")
