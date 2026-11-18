# Data formats

Every multi-byte value is little-endian.

## Model image

Written by `host/build_image.py`, read by `fw/model/model.c`
(`model_open`). The image is laid out exactly as it sits in DDR3 at offset
0. Every section starts on a 4 KB boundary, which the weight reader needs.

**Header** (4 KB, `ImageHeader` in `fw/model/model.h`), as consecutive
32-bit fields after the magic:

| Field | Meaning |
|---|---|
| `magic[8]` | `"LLMMODEL"` |
| `version` | format version, 1 |
| `total_bytes` | image size |
| `dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, gs` | the model config; gs (group size) must be 32 for the current build |
| `off_rms_att, off_rms_ffn, off_rms_final` | RMSNorm weights |
| `off_emb_q, off_emb_s` | token embedding values and scales |
| `off_tok, tok_bytes` | tokenizer |
| `n_mats, reserved` | number of matrices |
| `mat_off[n_mats]` | each matrix's offset: per layer wq wk wv wo w1 w2 w3, then the classifier |

**Sections:**

| Section | Type | Shape |
|---|---|---|
| rms_att | float32 | [n_layers][dim] |
| rms_ffn | float32 | [n_layers][dim] |
| rms_final | float32 | [dim] |
| emb_q | int8 | [vocab_size][dim] (Q8_0 token embedding) |
| emb_s | float32 | [vocab_size · dim / gs] |
| tokenizer | bytes | llama2.c `tokenizer.bin`, verbatim |
| matrices | qmv DDR format | see below; for stories15M the classifier is the token embedding (shared weights, as in `runq.c`), stored again in this layout |

For stories15M, the image is 28 MB and the matrices are 17.1 MB.

## qmv DDR format

This is how every matrix is stored, and what `wstream_reader` reads. It is
produced by `ddr_format()` in `host/llm_link.py`.

1. **Order the beats as the engine consumes them**
   ([qmv.md](hardware/qmv.md#weight-stream-order)): in blocks of 16 rows;
   per group; per row of the block; the group's 32-byte beat(s).
2. **Cut that sequence into chunks of 8 beats.** Each chunk is stored as 9
   AXI beats of 32 bytes:
   - AXI beat 0: the 8 beats' group scales (float32; scale j at bytes
     4j..4j+3);
   - AXI beats 1-8: the 8 beats' weights, 32 int8 each.

A matrix of B engine beats (B a multiple of 8) takes B/8 × 9 × 32 bytes. Its
start must be 4 KB aligned.

## microSD card

Raw blocks, with no file system ([hardware/sd_ctrl.md](hardware/sd_ctrl.md)):

| Block | Contents |
|---|---|
| 0 | `"LLMSDIMG"`, `version:u32` = 1, `nbytes:u32`, `ddr_addr:u32`, `checksum:u32` (the u32 sum of the image's words), then zeros |
| 1.. | the image, the last block zero-padded |

`image_tool.py --store` stores the model image with `ddr_addr` = 0.

## KV cache

The cache is in DDR3 from offset `0x2000_0000` (512 MB), laid out by
`fw/llm/attn_hw.c`. Each layer l and KV head h gets a K slot and a V slot,
each 4 KB aligned:

```
k_slot = align4k(ceil(seq_len / 128) * 128 * hs * 4)
v_slot = align4k(seq_len * hs * 4)
K(l, h) = 0x2000_0000 + (l * n_kv_heads + h) * k_slot
V(l, h) = 0x2000_0000 + n_layers * n_kv_heads * k_slot + (l * n_kv_heads + h) * v_slot
```

| Cache | Layout | Float address within the slot | Written by |
|---|---|---|---|
| K | blocks of 128 timesteps, each block `[i][t % 128]` | `(t / 128) * hs * 128 + i * 128 + t % 128` | DMA with stride 512 bytes, one word per element |
| V | rows `[t][i]` | `t * hs + i` | DMA with stride 4 (whole beats) |

## Q8_0 checkpoint (`runq.c` "version 2")

`ref/export_q80.py` writes this format from an fp32 llama2.c checkpoint,
reproducing llama2.c's `export.py version2_export()`. The format is:
1. a 256-byte header: magic `"ak42"`, version 2, the config, a
   shared-classifier flag and the group size;
2. the fp32 RMSNorm weights;
3. each tensor as int8 values followed by float32 group scales.

The group size is 64 unless dim isn't a multiple of it. For stories15M,
dim = 288, so the group size is 32. `host/build_image.py` reads this file.

## Reference dumps (`ref/build/<model>/`)

`ref/runq_dump` writes:

| File | Contents |
|---|---|
| `tokens.txt` | one token id per line: the prompt, then the generated tokens |
| `logits.bin` | float32 [steps][vocab_size], the logits after each `forward()` |
| `mm_NNN.bin` | every `matmul()` call at one position, in call order (43 for stories15M: 6 layers × 7, plus the classifier). `read_matmul()` in `ref/q80.py` reads them |

`fw/build/golden` and `ref/build/llm_host` write `tokens.txt` and
`logits.bin` in the same format.

## Trace lines

The board (`/trace`, `chat.py --trace`) and `fw/build/golden` print one
line per step:

```
T <pos> <token> <hash>
```

`hash` is the 32-bit FNV-1a hash (8 hex digits) over the bytes of all
`vocab_size` float32 logits of that step, least significant byte first
(`fw/model/trace.h`). Matching
lines mean bit-identical logits.
