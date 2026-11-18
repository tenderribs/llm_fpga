"""
Build a model image: everything top_soc's firmware needs to run a model,
laid out exactly as it sits in DDR3 (the image goes to DDR offset 0, from
the microSD card at boot or over the UART with host/image_tool.py).

Layout (all little endian; every section starts on a 4 KB boundary, which
the matmul engine needs for its weights):

  0x0000  header (4 KB), see fw/model/model.h ImageHeader:
            "LLMMODEL", version, total bytes,
            dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size,
            seq_len, group size,
            offsets of: rms_att, rms_ffn, rms_final, emb_q, emb_s, tokenizer;
            tokenizer bytes, number of matrices, then each matrix's offset
            (per layer wq wk wv wo w1 w2 w3, then the classifier)
          rms_att   float32 [n_layers][dim]
          rms_ffn   float32 [n_layers][dim]
          rms_final float32 [dim]
          emb_q     int8 [vocab_size][dim]      token embedding, Q8_0
          emb_s     float32 [vocab_size * dim / gs]
          tokenizer llama2.c tokenizer.bin, verbatim
          matrices  each in qmv DDR format (host/llm_link.py ddr_format)

usage: python3 host/build_image.py models/stories15M_q80.bin \
           third_party/llama2.c/tokenizer.bin models/stories15M.img
"""

import argparse as ap
from pathlib import Path
import struct
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm_link  # noqa: E402
from llm_link import q80  # noqa: E402

MAGIC = b"LLMMODEL"
VERSION = 1
HEADER_BYTES = 4096
ALIGN = 4096


def build(checkpoint: str, tokenizer: str) -> bytes:
    m = q80.Model(checkpoint)
    if m.gs != llm_link.GS_HW:
        sys.exit(f"group size {m.gs}: the engine is built for {llm_link.GS_HW}")
    L, dim, hid, kv = m.n_layers, m.dim, m.hidden_dim, m.kv_dim

    sections = []  # (bytes) in order, after the header
    offsets = []
    pos = HEADER_BYTES

    def add(data: bytes) -> int:
        nonlocal pos
        off = pos
        sections.append(data)
        offsets.append(off)
        pos += (len(data) + ALIGN - 1) // ALIGN * ALIGN
        return off

    off_rms_att = add(m.rms_att.astype("<f4").tobytes())
    off_rms_ffn = add(m.rms_ffn.astype("<f4").tobytes())
    off_rms_final = add(m.rms_final.astype("<f4").tobytes())
    off_emb_q = add(m.q_tokens.q.tobytes())
    off_emb_s = add(m.q_tokens.s.astype("<f4").tobytes())
    tok = Path(tokenizer).read_bytes()
    off_tok = add(tok)

    mats = []
    for l in range(L):
        for w, n, d in ((m.wq[l], dim, dim), (m.wk[l], dim, kv), (m.wv[l], dim, kv),
                        (m.wo[l], dim, dim), (m.w1[l], dim, hid), (m.w2[l], hid, dim),
                        (m.w3[l], dim, hid)):
            mats.append(add(llm_link.ddr_format(w, n, d, m.gs)))
    mats.append(add(llm_link.ddr_format(m.wcls, dim, m.vocab_size, m.gs)))

    header = MAGIC + struct.pack(
        "<II8i9I", VERSION, pos,
        dim, hid, L, m.n_heads, m.n_kv_heads, m.vocab_size, m.seq_len, m.gs,
        off_rms_att, off_rms_ffn, off_rms_final, off_emb_q, off_emb_s, off_tok, len(tok),
        len(mats), 0) + struct.pack(f"<{len(mats)}I", *mats)
    assert len(header) <= HEADER_BYTES, "too many matrices for the header"

    img = bytearray(pos)
    img[:len(header)] = header
    for off, data in zip(offsets, sections):
        img[off:off + len(data)] = data
    return bytes(img)


def main() -> int:
    parser = ap.ArgumentParser(prog="build_image.py", description=__doc__.split("\n")[1])
    parser.add_argument("checkpoint", help="Q8_0 checkpoint (ref/export_q80.py)")
    parser.add_argument("tokenizer", help="llama2.c tokenizer.bin")
    parser.add_argument("out")
    args = parser.parse_args()
    img = build(args.checkpoint, args.tokenizer)
    Path(args.out).write_bytes(img)
    print(f"wrote {args.out}: {len(img) / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
