"""
Quantize a llama2.c fp32 checkpoint (legacy "version 0" .bin, e.g. stories15M.bin)
to the Q8_0 "version 2" format that llama2.c's runq.c reads.

This is a numpy port of version2_export() in third_party/llama2.c/export.py, so
it works without PyTorch. It reproduces that function's output: same layout,
same group size backoff, same round-half-to-even weight rounding.

usage: python3 ref/export_q80.py models/stories15M.bin models/stories15M_q80.bin
"""

import argparse as ap
import struct

import numpy as np


def read_legacy(path: str):
    """Read a llama2.c legacy fp32 checkpoint (see run.c memory_map_weights)."""
    with open(path, "rb") as f:
        hdr = struct.unpack("7i", f.read(28))
        data = np.fromfile(f, dtype="<f4")

    dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len = hdr
    shared = vocab_size > 0
    vocab_size = abs(vocab_size)
    head_size = dim // n_heads
    kv_dim = n_kv_heads * head_size
    cfg = dict(dim=dim, hidden_dim=hidden_dim, n_layers=n_layers, n_heads=n_heads,
               n_kv_heads=n_kv_heads, vocab_size=vocab_size, seq_len=seq_len)

    off = 0

    def take(*shape):
        nonlocal off
        n = int(np.prod(shape))
        arr = data[off:off + n].reshape(shape)
        off += n
        return arr

    L = n_layers
    w = {}
    w["tok_embeddings"] = take(vocab_size, dim)
    w["rms_att"] = take(L, dim)
    w["wq"] = take(L, dim, dim)
    w["wk"] = take(L, kv_dim, dim)
    w["wv"] = take(L, kv_dim, dim)
    w["wo"] = take(L, dim, dim)
    w["rms_ffn"] = take(L, dim)
    w["w1"] = take(L, hidden_dim, dim)
    w["w2"] = take(L, dim, hidden_dim)
    w["w3"] = take(L, hidden_dim, dim)
    w["rms_final"] = take(dim)
    take(seq_len, head_size // 2)  # freq_cis_real: unused, runq computes RoPE
    take(seq_len, head_size // 2)  # freq_cis_imag
    w["output"] = w["tok_embeddings"] if shared else take(vocab_size, dim)
    assert off == data.size, f"{data.size - off} unread floats: wrong config?"
    return cfg, shared, w


def quantize_q80(w: np.ndarray, gs: int):
    """Symmetric int8 per group of gs, as export.py quantize_q80()."""
    w = w.astype(np.float32).reshape(-1, gs)
    scale = np.abs(w).max(axis=1) / np.float32(127.0)
    q = np.round(w / scale[:, None]).astype(np.int8)  # half to even, as torch.round
    err = np.abs(q.astype(np.float32) * scale[:, None] - w).max()
    return q.reshape(-1), scale.astype(np.float32), err


def export(cfg, shared, w, out_path: str, group_size: int = 64):
    while cfg["dim"] % group_size != 0:
        group_size //= 2
    L = cfg["n_layers"]

    quantized = [w["tok_embeddings"]]
    for name in ["wq", "wk", "wv", "wo", "w1", "w2", "w3"]:
        quantized += [w[name][l] for l in range(L)]
    if not shared:
        quantized.append(w["output"])

    with open(out_path, "wb") as f:
        f.write(struct.pack("I", 0x616B3432))  # "ak42"
        f.write(struct.pack("i", 2))  # version
        f.write(struct.pack("7i", cfg["dim"], cfg["hidden_dim"], L, cfg["n_heads"],
                            cfg["n_kv_heads"], cfg["vocab_size"], cfg["seq_len"]))
        f.write(struct.pack("B", int(shared)))
        f.write(struct.pack("i", group_size))
        f.write(b"\0" * (256 - f.tell()))

        for l in range(L):
            f.write(w["rms_att"][l].astype("<f4").tobytes())
        for l in range(L):
            f.write(w["rms_ffn"][l].astype("<f4").tobytes())
        f.write(w["rms_final"].astype("<f4").tobytes())

        max_err = 0.0
        for t in quantized:
            q, s, err = quantize_q80(t, group_size)
            f.write(q.tobytes())
            f.write(s.astype("<f4").tobytes())
            max_err = max(max_err, float(err))

    print(f"wrote {out_path}: group size {group_size}, "
          f"{len(quantized)} tensors, max group error {max_err:.6f}")


if __name__ == "__main__":
    parser = ap.ArgumentParser(prog="export_q80.py", description=__doc__.split("\n")[1])
    parser.add_argument("src", help="llama2.c legacy fp32 checkpoint")
    parser.add_argument("dst", help="output Q8_0 checkpoint for runq.c")
    parser.add_argument("--group-size", type=int, default=64)
    args = parser.parse_args()
    cfg, shared, w = read_legacy(args.src)
    print(cfg, "shared classifier" if shared else "separate classifier")
    export(cfg, shared, w, args.dst, args.group_size)
