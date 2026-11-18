# microSD controller (`sd_ctrl`)

Source: `src/sources/sd_ctrl.sv`. Tools: `host/sd_tool.py`,
`host/image_tool.py --store`.

`sd_ctrl` copies an image between the microSD card (SPI mode) and DDR3:

| Operation | Direction | Steps |
|---|---|---|
| **LOAD** (`load_i`) | card → DDR3 | reads the header in block 0, then the image from block 1 to the header's DDR3 address, and checks the checksum |
| **STORE** (`store_i`) | DDR3 → card | writes `store_bytes_i` bytes from `store_addr_i` to blocks 1 and up, then the header to block 0 |

STORE writes the header last, so an interrupted store never leaves a card
that looks valid. `llm_core` starts a LOAD by itself once DDR3 has
calibrated, so the model is in DDR3 about 7 s after configuration without
anyone asking.

## Card layout

Block 0 holds the header; the rest of the block is zero:

| Bytes | Field | Value |
|---|---|---|
| 0-7 | magic | `"LLMSDIMG"` |
| 8-11 | version | 1 |
| 12-15 | nbytes | image length, a multiple of 32 |
| 16-19 | ddr_addr | where the image belongs in DDR3, a multiple of 32 |
| 20-23 | checksum | sum of the image's little-endian u32 words, mod 2³² |

Blocks 1 and up hold the image, with the last block padded with zeros. The
card is used raw, with no file system. Storing an image overwrites whatever
the card held.

## Protocol

Every operation:
1. power-cycles the card through the slot's power switch (50 ms off, then
   50 ms on);
2. runs the SPI-mode init at about 400 kHz: CMD0, CMD8, ACMD41, CMD58, and
   CMD16 on byte-addressed (SDSC) cards;
3. switches to the fast clock: 22.5 MHz at 225 MHz (`FastHalf` = 5).

Reads use CMD18 (multi-block). They are not stopped with CMD12; the card is
simply deselected, and the next operation power-cycles it anyway. Writes go
one block at a time, with the data response and busy checks.

Both SDHC/SDXC (block-addressed) and SDSC (byte-addressed) cards work.

## Status and errors

When an operation finishes, `ok_o` or `err_o` stays set until the next one.
`img_bytes_o` and `img_addr_o` hold the header of the last successful
operation. `llm_core` reports all of this in STATUS: bits 1-3 and the error
code, through the UART or the CPU registers.

| Code | Meaning |
|---|---|
| 1 | CMD0 failed (no card, or no response) |
| 2 | CMD8 |
| 3 | ACMD41 (init timeout, ~1.4 s at 400 kHz) |
| 4 | CMD58 |
| 5 | CMD16 |
| 6 | read or write command rejected |
| 7 | data token timeout, or an error token |
| 8 | bad header (for example a blank card, or one that never had an image stored) |
| 9 | checksum mismatch |
| 10 | write data rejected |
| 11 | write busy timeout |

LED 3 blinks while an operation runs and is on after success. After an
error, LEDs 7:4 show the code.

## Pins

| Signal | SD pin (SPI mode) | Note |
|---|---|---|
| `sd_cs_o` | DAT3 | |
| `sd_mosi_o` | CMD | |
| `sd_miso_i` | DAT0 | |
| `sd_sclk_o` | CLK | |
| `sd_power_n_o` | slot power switch | Genesys 2 `sd_reset`; low = on |
| `sd_cd_i` | card detect | reported in the UART STATUS flags, bit 4 |

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `ClkHz` | 225 000 000 | clock |
| `SlowHalf` | `ClkHz / 800 000` | half SCLK period during init (~400 kHz) |
| `FastHalf` | 5 | half SCLK period afterwards (22.5 MHz) |
| `PowerClks` | `ClkHz / 20` | power-cycle time (50 ms) |
| `AddrW` | 30 | DDR3 address width |

The simulation shortens these.

## Verification

- `tb_sd_ctrl` (`sim/run_tb.sh tb_sd_ctrl sdhc -GSdhc=1`, and `-GSdhc=0`
  for SDSC) runs a store and a load against `src/sim/sd_card_model.sv`. The
  model is an SPI-mode card with random delays that also flags protocol
  violations.
- `tb_llm_core` section 5 does an SD round trip of a matmul's weights.
