`default_nettype none

// microSD card (SPI mode) <-> DDR image copier.
//
//   LOAD  (load_i):  card -> DDR. Reads the header in block 0, then the image
//                    from block 1 into DDR at the header's address, and checks
//                    the checksum.
//   STORE (store_i): DDR -> card. Writes store_bytes_i bytes from DDR at
//                    store_addr_i to blocks 1.., then the header to block 0.
//                    The header goes last, so an interrupted store never
//                    leaves a card that looks valid.
//
// Card layout, block 0 (rest of the block zero):
//   "LLMSDIMG"     magic
//   version  u32  1
//   nbytes   u32  image length in bytes, multiple of 32
//   ddr_addr u32  where the image belongs in DDR, multiple of 32
//   checksum u32  sum of the image's little-endian u32 words, mod 2^32
// blocks 1..: the image, the last block padded with zeros.
//
// Every operation starts by power-cycling the card and running the SPI-mode
// init (CMD0, CMD8, ACMD41, CMD58, CMD16 on byte-addressed cards) at
// ~400 kHz, then switches to the fast clock. Multi-block reads (CMD18) are not
// stopped with CMD12: the card is just deselected, and the next operation
// power-cycles it anyway.
//
// When done, ok_o or err_o is held until the next operation. err_code_o:
//   1 CMD0  2 CMD8  3 ACMD41  4 CMD58  5 CMD16  6 read/write command rejected
//   7 data token timeout / error token  8 bad header  9 checksum mismatch
//   10 write data rejected  11 write busy timeout
// img_bytes_o / img_addr_o: the header of the last successful operation.
//
// SD pins (SPI mode): sd_cs_o = DAT3, sd_mosi_o = CMD, sd_miso_i = DAT0.
// sd_power_n_o drives the slot's power switch (Genesys 2 sd_reset): low = on.
module sd_ctrl #(
    parameter int ClkHz = 225_000_000,
    parameter int SlowHalf = ClkHz / 800_000,  // half SCLK period, ~400 kHz
    parameter int FastHalf = 5,  // 22.5 MHz at 225 MHz
    parameter int PowerClks = ClkHz / 20,  // 50 ms off, then 50 ms on
    parameter int AddrW = 30
) (
    input wire logic clk_i,
    input wire logic rst_ni,
    input wire logic load_i,
    input wire logic store_i,
    input wire logic [31:0] store_addr_i,
    input wire logic [31:0] store_bytes_i,
    output logic busy_o,
    output logic ok_o,
    output logic err_o,
    output logic [3:0] err_code_o,
    output logic [31:0] img_bytes_o,
    output logic [31:0] img_addr_o,
    output logic [7:0] last_r1_o,  // last R1 response (debug)

    output logic sd_power_n_o,
    output logic sd_sclk_o,
    output logic sd_cs_o,
    output logic sd_mosi_o,
    input wire logic sd_miso_i,

    // AXI4, single beats: writes for LOAD, reads for STORE
    output logic [AddrW-1:0] m_awaddr_o,
    output logic m_awvalid_o,
    input wire logic m_awready_i,
    output logic [255:0] m_wdata_o,
    output logic m_wvalid_o,
    input wire logic m_wready_i,
    input wire logic m_bvalid_i,
    output logic m_bready_o,
    output logic [AddrW-1:0] m_araddr_o,
    output logic m_arvalid_o,
    input wire logic m_arready_i,
    input wire logic [255:0] m_rdata_i,
    input wire logic m_rvalid_i,
    output logic m_rready_o
);

  // ---------------------------------------------------------------------------
  // SPI byte engine, mode 0: MOSI changes while SCLK is low, both sides sample
  // on the rising edge. MISO goes through a 2-flop synchronizer and is read at
  // the end of the high phase, so the half period must be >= 3 cycles.
  // ---------------------------------------------------------------------------
  (* ASYNC_REG = "TRUE" *) logic [1:0] miso_sync_q;
  logic miso;
  assign miso = miso_sync_q[1];

  logic x_start, x_busy_q, x_done;
  logic [7:0] x_tx, x_shift_q, x_rx_q;
  logic [2:0] x_bit_q;
  logic x_high_q;
  logic [15:0] x_cnt_q, half;
  logic fast_q;

  assign half = fast_q ? 16'(FastHalf) : 16'(SlowHalf);

  always_ff @(posedge clk_i, negedge rst_ni) begin
    if (~rst_ni) begin
      miso_sync_q <= '1;
      x_busy_q <= 1'b0;
      x_done <= 1'b0;
      x_bit_q <= '0;
      x_high_q <= 1'b0;
      x_cnt_q <= '0;
      x_shift_q <= '1;
      x_rx_q <= '1;
      sd_sclk_o <= 1'b0;
      sd_mosi_o <= 1'b1;
    end else begin
      miso_sync_q <= {miso_sync_q[0], sd_miso_i};
      x_done <= 1'b0;
      if (!x_busy_q) begin
        if (x_start) begin
          x_busy_q <= 1'b1;
          x_shift_q <= x_tx;
          sd_mosi_o <= x_tx[7];
          x_bit_q <= 3'd7;
          x_high_q <= 1'b0;
          x_cnt_q <= half - 1;
        end
      end else if (x_cnt_q != 0) begin
        x_cnt_q <= x_cnt_q - 1;
      end else begin
        x_cnt_q <= half - 1;
        if (!x_high_q) begin
          sd_sclk_o <= 1'b1;
          x_high_q <= 1'b1;
        end else begin
          // end of the high phase: take the bit, SCLK low, next MOSI bit
          sd_sclk_o <= 1'b0;
          x_high_q <= 1'b0;
          x_rx_q <= {x_rx_q[6:0], miso};
          if (x_bit_q == 0) begin
            x_busy_q <= 1'b0;
            x_done <= 1'b1;
            sd_mosi_o <= 1'b1;
          end else begin
            x_bit_q <= x_bit_q - 1;
            x_shift_q <= {x_shift_q[6:0], 1'b1};
            sd_mosi_o <= x_shift_q[6];
          end
        end
      end
    end
  end

  // ---------------------------------------------------------------------------
  // Control
  // ---------------------------------------------------------------------------
  typedef enum logic [5:0] {
    S_IDLE,
    S_PWR_OFF,
    S_PWR_ON,
    S_WAKE,        // >= 74 clocks with CS high
    // generic command
    S_CMD,         // 6 command bytes
    S_R1,          // wait for R1 (MSB clear)
    S_REXTRA,      // 4 more response bytes (R3 / R7)
    S_CMD_END,     // deselect + one byte of clocks, unless keep_cs
    // init sequence: each state checks the last response
    S_CMD0,
    S_CMD8,
    S_CMD55,
    S_ACMD41,
    S_CMD58,
    S_CMD16,
    S_READY,       // card initialized at full speed: start the operation
    // LOAD
    S_RD_HDR,      // CMD17 block 0 answered
    S_RD_IMG,      // CMD18 answered
    S_TOKEN,       // wait for the 0xFE start token
    S_RDATA,       // 512 data bytes
    S_RCRC,        // 2 CRC bytes, ignored
    S_AXI_WR,      // a 32-byte beat to DDR
    S_CHECK_HDR,
    S_RD_FINISH,
    // STORE
    S_WR_IMG,      // CMD25 answered
    S_WR_HDR,      // CMD24 answered
    S_WGAP,        // one byte before the data token
    S_WTOKEN,      // 0xFC (multi) / 0xFE (single)
    S_WDATA,       // 512 data bytes
    S_AXI_RD,      // fetch the next 32-byte beat from DDR
    S_WCRC,        // 2 dummy CRC bytes
    S_WRESP,       // data response token
    S_WBUSY,       // wait while the card programs
    S_WSTOP,       // 0xFD stop token after the last multi-block block
    S_WSTOP_GAP,
    S_WR_FINISH,
    S_WR_HDR_CMD,  // issue CMD24 for the header block
    S_DONE
  } state_t;
  state_t state_q, ret_q;

  logic store_q;  // the current operation is STORE
  logic [47:0] cmd_q;
  logic [2:0] cmd_byte_q;
  logic want_extra_q, keep_cs_q, end_started_q;
  logic [7:0] r1_q;
  logic [31:0] rextra_q;
  logic [4:0] tries_q;
  logic [31:0] timer_q;
  logic sdhc_q, v2_q;
  logic hdr_q;  // current block is the header (block 0)
  logic multi_q;  // current write is part of the multi-block write
  logic [9:0] byte_idx_q;
  logic [31:0] img_left_q;  // image bytes still to move
  logic [31:0] sum_q, word_q;
  logic [4:0] beat_byte_q;
  logic beat_valid_q;  // STORE: beat_q holds unsent DDR data
  logic [31:0][7:0] beat_q;
  logic [AddrW-1:0] addr_q;
  logic aw_done_q, w_done_q;
  logic [3:0] err_q;

  logic [63:0] magic_q;
  logic [31:0] version_q, nbytes_q, ddr_addr_q, checksum_q;

  assign busy_o = (state_q != S_IDLE) && (state_q != S_DONE);
  assign ok_o = (state_q == S_DONE) && (err_q == 0);
  assign err_o = (state_q == S_DONE) && (err_q != 0);
  assign err_code_o = err_q;
  assign img_bytes_o = nbytes_q;
  assign img_addr_o = ddr_addr_q;
  assign last_r1_o = r1_q;

  assign m_awaddr_o = addr_q;
  assign m_awvalid_o = (state_q == S_AXI_WR) && !aw_done_q;
  assign m_wdata_o = beat_q;
  assign m_wvalid_o = (state_q == S_AXI_WR) && !w_done_q;
  assign m_bready_o = (state_q == S_AXI_WR);
  assign m_araddr_o = addr_q;
  assign m_arvalid_o = (state_q == S_AXI_RD) && !aw_done_q;  // aw_done_q reused: AR sent
  assign m_rready_o = (state_q == S_AXI_RD);

  function automatic logic [47:0] frame(input logic [5:0] idx, input logic [31:0] arg,
                                        input logic [7:0] crc);
    return {2'b01, idx, arg, crc};
  endfunction

  // header block byte i (STORE)
  function automatic logic [7:0] hdr_byte(input logic [9:0] i, input logic [31:0] nbytes,
                                          input logic [31:0] addr, input logic [31:0] sum);
    logic [63:0] magic = "LLMSDIMG";
    if (i < 8) return magic[63-8*i-:8];
    if (i < 12) return (i == 8) ? 8'd1 : 8'd0;
    if (i < 16) return nbytes[8*(i-12)+:8];
    if (i < 20) return addr[8*(i-16)+:8];
    if (i < 24) return sum[8*(i-20)+:8];
    return 8'd0;
  endfunction

  // STORE: the next data byte
  logic need_fetch;
  logic [7:0] wbyte;
  assign need_fetch = !hdr_q && (img_left_q != 0) && !beat_valid_q;
  always_comb begin
    if (hdr_q) wbyte = hdr_byte(byte_idx_q, nbytes_q, ddr_addr_q, sum_q);
    else if (img_left_q != 0) wbyte = beat_q[beat_byte_q];
    else wbyte = 8'd0;  // padding after the image
  end

  // byte transfers: started combinationally whenever the state wants one
  always_comb begin
    x_start = 1'b0;
    x_tx = 8'hFF;
    if (!x_busy_q && !x_done) begin
      case (state_q)
        S_WAKE, S_R1, S_REXTRA, S_TOKEN, S_RDATA, S_RCRC, S_WGAP, S_WCRC, S_WRESP, S_WBUSY,
        S_WSTOP_GAP:
        x_start = 1'b1;
        S_CMD: begin
          x_start = 1'b1;
          x_tx = cmd_q[47-cmd_byte_q*8-:8];
        end
        S_CMD_END: x_start = !keep_cs_q && !end_started_q;
        S_WTOKEN: begin
          x_start = 1'b1;
          x_tx = multi_q ? 8'hFC : 8'hFE;
        end
        S_WDATA: begin
          x_start = !need_fetch;
          x_tx = wbyte;
        end
        S_WSTOP: begin
          x_start = 1'b1;
          x_tx = 8'hFD;
        end
        default: ;
      endcase
    end
  end

  task automatic issue(input logic [47:0] c, input logic extra, input logic keep_cs,
                       input state_t ret);
    cmd_q <= c;
    cmd_byte_q <= '0;
    want_extra_q <= extra;
    keep_cs_q <= keep_cs;
    end_started_q <= 1'b0;
    ret_q <= ret;
    tries_q <= '0;
    sd_cs_o <= 1'b0;
    state_q <= S_CMD;
  endtask

  // deselect and give the card 8 clocks to finish, then continue at `next`
  // (the card needs them after a data transfer before the next command)
  task automatic release_then(input state_t next);
    keep_cs_q <= 1'b0;
    end_started_q <= 1'b0;
    ret_q <= next;
    sd_cs_o <= 1'b1;
    state_q <= S_CMD_END;
  endtask

  task automatic fail(input logic [3:0] code);
    err_q <= code;
    sd_cs_o <= 1'b1;
    state_q <= S_DONE;
  endtask

  // block address argument for block b
  function automatic logic [31:0] blk_arg(input logic sdhc, input logic [31:0] b);
    return sdhc ? b : b * 512;
  endfunction

  always_ff @(posedge clk_i, negedge rst_ni) begin
    if (~rst_ni) begin
      state_q <= S_IDLE;
      ret_q <= S_IDLE;
      store_q <= 1'b0;
      sd_power_n_o <= 1'b1;
      sd_cs_o <= 1'b1;
      fast_q <= 1'b0;
      err_q <= '0;
      timer_q <= '0;
      tries_q <= '0;
      cmd_byte_q <= '0;
      end_started_q <= 1'b0;
      keep_cs_q <= 1'b0;
      aw_done_q <= 1'b0;
      w_done_q <= 1'b0;
      nbytes_q <= '0;
      ddr_addr_q <= '0;
      hdr_q <= 1'b0;
      multi_q <= 1'b0;
      beat_valid_q <= 1'b0;
    end else begin
      if (state_q == S_CMD_END && x_start) end_started_q <= 1'b1;

      case (state_q)
        S_IDLE, S_DONE: begin
          if (load_i || store_i) begin
            store_q <= store_i;
            if (store_i) begin
              nbytes_q <= store_bytes_i;
              ddr_addr_q <= store_addr_i;
            end
            err_q <= '0;
            fast_q <= 1'b0;
            sd_cs_o <= 1'b1;
            sd_power_n_o <= 1'b1;  // power off
            timer_q <= '0;
            state_q <= S_PWR_OFF;
          end
        end

        S_PWR_OFF: begin
          timer_q <= timer_q + 1;
          if (timer_q == 32'(PowerClks)) begin
            sd_power_n_o <= 1'b0;  // power on
            timer_q <= '0;
            state_q <= S_PWR_ON;
          end
        end

        S_PWR_ON: begin
          timer_q <= timer_q + 1;
          if (timer_q == 32'(PowerClks)) begin
            tries_q <= '0;
            state_q <= S_WAKE;
          end
        end

        S_WAKE: begin
          if (x_done) begin
            tries_q <= tries_q + 1;
            if (tries_q == 9) issue(frame(6'd0, 32'd0, 8'h95), 1'b0, 1'b0, S_CMD0);
          end
        end

        // ---- generic command
        S_CMD: begin
          if (x_done) begin
            cmd_byte_q <= cmd_byte_q + 1;
            if (cmd_byte_q == 3'd5) begin
              tries_q <= '0;
              state_q <= S_R1;
            end
          end
        end

        S_R1: begin
          if (x_done) begin
            tries_q <= tries_q + 1;
            if (!x_rx_q[7]) begin
              r1_q <= x_rx_q;
              tries_q <= '0;
              state_q <= want_extra_q ? S_REXTRA : S_CMD_END;
            end else if (tries_q == 5'd16) begin
              r1_q <= 8'hFF;  // no response
              state_q <= S_CMD_END;
            end
          end
        end

        S_REXTRA: begin
          if (x_done) begin
            rextra_q <= {rextra_q[23:0], x_rx_q};
            tries_q <= tries_q + 1;
            if (tries_q == 5'd3) state_q <= S_CMD_END;
          end
        end

        S_CMD_END: begin
          if (keep_cs_q) begin
            state_q <= ret_q;
          end else begin
            sd_cs_o <= 1'b1;
            if (x_done) state_q <= ret_q;
          end
        end

        // ---- init
        S_CMD0: begin
          if (r1_q != 8'h01) fail(4'd1);
          else issue(frame(6'd8, 32'h0000_01AA, 8'h87), 1'b1, 1'b0, S_CMD8);
        end

        S_CMD8: begin
          timer_q <= '0;
          if (r1_q == 8'h05) begin  // illegal command: SD v1 card
            v2_q <= 1'b0;
            issue(frame(6'd55, 32'd0, 8'h01), 1'b0, 1'b0, S_CMD55);
          end else if (r1_q != 8'h01 || rextra_q[11:0] != 12'h1AA) begin
            fail(4'd2);
          end else begin
            v2_q <= 1'b1;
            issue(frame(6'd55, 32'd0, 8'h01), 1'b0, 1'b0, S_CMD55);
          end
        end

        S_CMD55: begin
          issue(frame(6'd41, v2_q ? 32'h4000_0000 : 32'd0, 8'h01), 1'b0, 1'b0, S_ACMD41);
        end

        S_ACMD41: begin
          timer_q <= timer_q + 1;
          if (r1_q == 8'h00) issue(frame(6'd58, 32'd0, 8'h01), 1'b1, 1'b0, S_CMD58);
          else if (timer_q > 32'd4000) fail(4'd3);  // 4000 tries: ~1.4 s at 400 kHz
          else issue(frame(6'd55, 32'd0, 8'h01), 1'b0, 1'b0, S_CMD55);
        end

        S_CMD58: begin
          if (r1_q != 8'h00) begin
            fail(4'd4);
          end else begin
            sdhc_q <= v2_q && rextra_q[30];  // CCS: block addressed
            if (v2_q && rextra_q[30]) state_q <= S_READY;
            else issue(frame(6'd16, 32'd512, 8'h01), 1'b0, 1'b0, S_CMD16);
          end
        end

        S_CMD16: begin
          if (r1_q != 8'h00) fail(4'd5);
          else state_q <= S_READY;
        end

        S_READY: begin
          fast_q <= 1'b1;
          if (store_q) begin
            // image to blocks 1.., header last
            hdr_q <= 1'b0;
            multi_q <= 1'b1;
            img_left_q <= nbytes_q;
            addr_q <= AddrW'(ddr_addr_q);
            sum_q <= '0;
            beat_byte_q <= '0;
            beat_valid_q <= 1'b0;
            issue(frame(6'd25, blk_arg(sdhc_q, 32'd1), 8'h01), 1'b0, 1'b1, S_WR_IMG);
          end else begin
            hdr_q <= 1'b1;
            issue(frame(6'd17, 32'd0, 8'h01), 1'b0, 1'b1, S_RD_HDR);
          end
        end

        // ---- LOAD
        S_RD_HDR, S_RD_IMG: begin
          if (r1_q != 8'h00) begin
            fail(4'd6);
          end else begin
            timer_q <= '0;
            state_q <= S_TOKEN;
          end
        end

        S_TOKEN: begin
          timer_q <= timer_q + 1;
          if (x_done) begin
            if (x_rx_q == 8'hFE) begin
              byte_idx_q <= '0;
              state_q <= S_RDATA;
            end else if (x_rx_q != 8'hFF) begin
              fail(4'd7);  // data error token
            end
          end else if (timer_q > 32'(ClkHz / 4)) begin
            fail(4'd7);
          end
        end

        S_RDATA: begin
          if (x_done) begin
            byte_idx_q <= byte_idx_q + 1;
            if (hdr_q) begin
              case (byte_idx_q)
                0, 1, 2, 3, 4, 5, 6, 7: magic_q <= {magic_q[55:0], x_rx_q};
                8, 9, 10, 11: version_q <= {x_rx_q, version_q[31:8]};
                12, 13, 14, 15: nbytes_q <= {x_rx_q, nbytes_q[31:8]};
                16, 17, 18, 19: ddr_addr_q <= {x_rx_q, ddr_addr_q[31:8]};
                20, 21, 22, 23: checksum_q <= {x_rx_q, checksum_q[31:8]};
                default: ;
              endcase
            end else if (img_left_q != 0) begin
              beat_q <= {x_rx_q, beat_q[31:1]};
              word_q <= {x_rx_q, word_q[31:8]};
              beat_byte_q <= beat_byte_q + 1;
              img_left_q <= img_left_q - 1;
              if (beat_byte_q[1:0] == 2'd3) sum_q <= sum_q + {x_rx_q, word_q[31:8]};
            end
            // a full beat goes to DDR first, even when it ends the block
            // (blocks are 16 beats, so the last beat always does)
            if (!hdr_q && img_left_q != 0 && beat_byte_q == 5'd31) begin
              aw_done_q <= 1'b0;
              w_done_q <= 1'b0;
              state_q <= S_AXI_WR;  // beat_q is complete next cycle
            end else if (byte_idx_q == 10'd511) begin
              tries_q <= '0;
              state_q <= S_RCRC;
            end
          end
        end

        S_AXI_WR: begin
          if (m_awvalid_o && m_awready_i) aw_done_q <= 1'b1;
          if (m_wvalid_o && m_wready_i) w_done_q <= 1'b1;
          if (m_bvalid_i) begin
            addr_q <= addr_q + 32;
            tries_q <= '0;
            state_q <= (byte_idx_q == 10'd512) ? S_RCRC : S_RDATA;
          end
        end

        S_RCRC: begin
          if (x_done) begin
            tries_q <= tries_q + 1;
            if (tries_q == 5'd1) begin
              if (hdr_q) begin
                release_then(S_CHECK_HDR);
              end else if (img_left_q == 0) begin
                state_q <= S_RD_FINISH;
              end else begin
                timer_q <= '0;
                state_q <= S_TOKEN;  // next block of the multi-block read
              end
            end
          end
        end

        S_CHECK_HDR: begin
          if (magic_q != "LLMSDIMG" || version_q != 1 || nbytes_q[4:0] != 0
              || ddr_addr_q[4:0] != 0 || nbytes_q == 0) begin
            fail(4'd8);
          end else begin
            hdr_q <= 1'b0;
            img_left_q <= nbytes_q;
            addr_q <= AddrW'(ddr_addr_q);
            sum_q <= '0;
            beat_byte_q <= '0;
            issue(frame(6'd18, blk_arg(sdhc_q, 32'd1), 8'h01), 1'b0, 1'b1, S_RD_IMG);
          end
        end

        S_RD_FINISH: begin
          sd_cs_o <= 1'b1;
          if (sum_q != checksum_q) fail(4'd9);
          else state_q <= S_DONE;
        end

        // ---- STORE
        S_WR_IMG, S_WR_HDR: begin
          if (r1_q != 8'h00) begin
            fail(4'd6);
          end else begin
            state_q <= S_WGAP;
          end
        end

        S_WGAP: begin
          if (x_done) state_q <= S_WTOKEN;
        end

        S_WTOKEN: begin
          if (x_done) begin
            byte_idx_q <= '0;
            state_q <= S_WDATA;
          end
        end

        S_WDATA: begin
          if (need_fetch && !x_busy_q && !x_done) begin
            aw_done_q <= 1'b0;  // AR not yet sent
            state_q <= S_AXI_RD;
          end
          if (x_done) begin
            byte_idx_q <= byte_idx_q + 1;
            if (!hdr_q && img_left_q != 0) begin
              beat_byte_q <= beat_byte_q + 1;
              img_left_q <= img_left_q - 1;
              if (beat_byte_q == 5'd31) begin
                beat_valid_q <= 1'b0;
                addr_q <= addr_q + 32;
              end
            end
            if (byte_idx_q == 10'd511) begin
              tries_q <= '0;
              state_q <= S_WCRC;
            end
          end
        end

        S_AXI_RD: begin
          if (m_arvalid_o && m_arready_i) aw_done_q <= 1'b1;
          if (m_rvalid_i) begin
            beat_q <= m_rdata_i;
            beat_valid_q <= 1'b1;
            beat_byte_q <= '0;
            state_q <= S_WDATA;
          end
        end

        S_WCRC: begin
          if (x_done) begin
            tries_q <= tries_q + 1;
            if (tries_q == 5'd1) state_q <= S_WRESP;
          end
        end

        S_WRESP: begin
          if (x_done) begin
            if (x_rx_q[4:0] != 5'b00101) begin
              fail(4'd10);
            end else begin
              timer_q <= '0;
              state_q <= S_WBUSY;
            end
          end
        end

        S_WBUSY: begin
          timer_q <= timer_q + 1;
          if (x_done && x_rx_q == 8'hFF) begin
            if (hdr_q) begin
              state_q <= S_WR_FINISH;
            end else if (!multi_q) begin
              // the stop token's busy period is over: write the header
              hdr_q <= 1'b1;
              release_then(S_WR_HDR_CMD);
            end else if (img_left_q == 0) begin
              state_q <= S_WSTOP;
            end else begin
              state_q <= S_WTOKEN;  // next block, no gap byte needed
            end
          end else if (!x_done && timer_q > 32'(ClkHz / 2)) begin
            fail(4'd11);
          end
        end

        S_WSTOP: begin
          if (x_done) begin
            multi_q <= 1'b0;
            state_q <= S_WSTOP_GAP;
          end
        end

        S_WSTOP_GAP: begin
          // one byte, then busy until the card has finished
          if (x_done) begin
            timer_q <= '0;
            state_q <= S_WBUSY;
          end
        end

        S_WR_HDR_CMD: begin
          issue(frame(6'd24, 32'd0, 8'h01), 1'b0, 1'b1, S_WR_HDR);
        end

        S_WR_FINISH: begin
          sd_cs_o <= 1'b1;
          state_q <= S_DONE;
        end

        default: state_q <= S_IDLE;
      endcase

      // STORE checksum: add up the image bytes as they are sent
      if (state_q == S_WDATA && x_start && !hdr_q && img_left_q != 0) begin
        word_q <= {x_tx, word_q[31:8]};
        if (beat_byte_q[1:0] == 2'd3) sum_q <= sum_q + {x_tx, word_q[31:8]};
      end
    end
  end

endmodule

`default_nettype wire
