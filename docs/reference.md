# Golden reference (`ref/`)

`ref/` defines "correct": Karpathy's llama2.c `runq.c`
(`third_party/llama2.c`), and a numpy reference that the test vectors are
built from.

## Build it

```sh
mkdir -p models && curl -L -o models/stories15M.bin \
  https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
ref/build_ref.sh ["prompt"] [steps] [dump_pos]     # defaults: "Once upon a time" 64 0
python3 ref/check_ref.py [ref/build/stories15M]
```

`ref/build_ref.sh` runs three steps:

1. **Quantize.** If `models/stories15M_q80.bin` is missing, it is made from
   the fp32 checkpoint with `export_q80.py`.
2. **Patch `runq.c`.** `runq.c`'s `matmul()` is renamed to `matmul_orig()`,
   giving `ref/build/runq_patched.c`. `runq_dump.c` includes that file and
   wraps `matmul()` so that it sees every call.
3. **Compile and run.** It is built with `gcc -O2`, without `-ffast-math`
   or OpenMP, so the float math runs in source order. It then runs
   greedily and writes `ref/build/stories15M/`:
   - `tokens.txt`;
   - `logits.bin`;
   - `mm_000.bin` ... `mm_042.bin`, the 43 matmul calls at `dump_pos`.

The file formats are in [data_formats.md](data_formats.md#reference-dumps-refbuildmodel).

## Files

| File | What |
|---|---|
| `export_q80.py` | numpy port of llama2.c's `export.py version2_export()`, so no PyTorch is needed. Same layout, same group-size backoff, same round-half-to-even weight rounding. Writes the Q8_0 "version 2" file `runq.c` reads. `--group-size` (default 64, halved until it divides dim); the board needs 32 |
| `runq_dump.c` | the dumping wrapper around `runq.c` (greedy, no sampling) |
| `q80.py` | numpy reference: Q8_0 tensors, `matmul()` and `quantize()` bit-exact with `runq.c`, `forward()`, `read_matmul()` for the dumps, `qmv_stream()` (the engine's weight order) |
| `check_ref.py` | checks `q80.py` against the dumps |
| `build_ref.sh` | the steps above |

## What is bit-exact and what isn't

**`q80.matmul()` and `q80.quantize()` are bit-exact** with `runq.c`. They
define what the matmul engine computes, and the engine's test vectors come
from them (`sim/gen_qmv.py`).

**`q80.forward()` is close but not exact.** numpy's float sums and its
`exp` and `sqrt` differ from C's by an ulp now and then. Once one tips an
activation across a quantization rounding boundary, an int8 differs by 1,
and the KV cache carries the difference forward. `check_ref.py` therefore
accepts two things:
- logits within 1.0;
- the same greedy token wherever `runq.c`'s top two logits are more than
  0.1 apart.

**For the whole model, `runq.c` itself is the golden.** The chain to the
board is:

```
ref/runq_dump  ==  fw/build/golden_libm   (the firmware's forward() with glibc math)
fw/build/golden_libm  ->  fw/build/golden (the firmware's own mathf.c)
fw/build/golden  ==  the board            (trace lines)
```

See [system_design.md](system_design.md#11-verification).

## Desktop baseline

For [performance.md](performance.md), `runq.c` was built three ways and run
on the same prompt:
- `gcc -O2` (the bit-exact build, same as `runq_dump`);
- `-Ofast -march=native`;
- `-Ofast -march=native -fopenmp`.
