`default_nettype none

// UART command interface to DDR and the qmv matmul engine. Everything runs on
// the DDR controller's user clock; DDR is reached through an AXI4 master port.
// Kept separate from the DDR controller (top_llm) so it can be simulated with
// an AXI memory model.
//
// Commands (8N1, Baud), all little-endian. Each starts with an opcode byte:
//   0x00 STATUS                          reply "LLM" + {7'b0, calib_done}
//   0x01 WRITE addr:u32 beats:u32 data   write beats*32 bytes at addr (32 B
//                                        aligned); reply 0x01 when written
//   0x02 READ  addr:u32 beats:u32        reply beats*32 bytes from addr
//   0x03 MATMUL addr:u32 n:u16 d:u16 x.q x.s
//        W (d x n) is at addr (4 KB aligned) in qmv DDR format (see
//        wstream_reader.sv); reply d float32 results, rows in order
//   0x04 SD_LOAD                         reply 0x04, then copy the microSD
//                                        image into DDR (see sd_ctrl.sv)
//   0x05 SD_STORE addr:u32 nbytes:u32    reply 0x05, then copy nbytes of DDR
//                                        at addr to the microSD card
// STATUS reply, 12 bytes:
//   "LLM", flags, sd_err_code, sd_last_r1, 0, 0, sd_image_bytes:u32
//   flags: 0 DDR calibrated, 1 SD busy, 2 SD ok, 3 SD error, 4 card detect pin
// SD_LOAD also runs by itself once the DDR has calibrated. While an SD
// operation runs only STATUS is answered; other commands are dropped.
// If the host stops partway through a command for IdleTimeoutClks, it is
// dropped and the next byte is read as an opcode.
module llm_core #(
    parameter int ClkHz = 225_000_000,
    parameter int Baud = 3_000_000,
    parameter int IdleTimeoutClks = ClkHz / 10,
    parameter int GS = 32,
    parameter int MaxN = 2048,
    parameter int AddrW = 30,
    // SD timing, see sd_ctrl.sv (overridden in simulation)
    parameter int SdSlowHalf = ClkHz / 800_000,
    parameter int SdFastHalf = 5,
    parameter int SdPowerClks = ClkHz / 20
) (
    input wire logic clk_i,
    input wire logic rst_ni,
    input wire logic calib_done_i,

    input wire logic uart_rx_i,
    output logic uart_tx_o,
    output logic [7:0] led_o,

    // AXI4 master, 256-bit data, ID 0
    output logic [AddrW-1:0] m_awaddr_o,
    output logic [7:0] m_awlen_o,
    output logic m_awvalid_o,
    input wire logic m_awready_i,
    output logic [255:0] m_wdata_o,
    output logic m_wlast_o,
    output logic m_wvalid_o,
    input wire logic m_wready_i,
    input wire logic m_bvalid_i,
    output logic m_bready_o,
    output logic [AddrW-1:0] m_araddr_o,
    output logic [7:0] m_arlen_o,
    output logic m_arvalid_o,
    input wire logic m_arready_i,
    input wire logic [255:0] m_rdata_i,
    input wire logic m_rvalid_i,
    output logic m_rready_o,

    // microSD, SPI mode (see sd_ctrl.sv)
    output logic sd_power_n_o,
    output logic sd_sclk_o,
    output logic sd_cs_o,
    output logic sd_mosi_o,
    input wire logic sd_miso_i,
    input wire logic sd_cd_i
);

  localparam int P = 32;

  // ---------------------------------------------------------------------------
  // AXI: the command logic (c_*) owns the port, except while the SD
  // controller (sd_*) is busy
  // ---------------------------------------------------------------------------
  logic [AddrW-1:0] c_awaddr, c_araddr, sd_awaddr, sd_araddr;
  logic [7:0] c_awlen, c_arlen;
  logic c_awvalid, c_awready, c_wlast, c_wvalid, c_wready, c_bvalid, c_bready;
  logic c_arvalid, c_arready, c_rvalid, c_rready;
  logic [255:0] c_wdata, c_rdata, sd_wdata;
  logic sd_awvalid, sd_wvalid, sd_bready, sd_arvalid, sd_rready;
  logic sd_busy, sd_ok, sd_err;
  logic [3:0] sd_err_code;
  logic [31:0] sd_img_bytes, sd_img_addr;
  logic [7:0] sd_last_r1;
  logic sd_load, sd_store;
  logic [31:0] sd_store_addr_q, sd_store_bytes_q;

  assign m_awaddr_o  = sd_busy ? sd_awaddr : c_awaddr;
  assign m_awlen_o   = sd_busy ? 8'd0 : c_awlen;
  assign m_awvalid_o = sd_busy ? sd_awvalid : c_awvalid;
  assign m_wdata_o   = sd_busy ? sd_wdata : c_wdata;
  assign m_wlast_o   = sd_busy ? 1'b1 : c_wlast;
  assign m_wvalid_o  = sd_busy ? sd_wvalid : c_wvalid;
  assign m_bready_o  = sd_busy ? sd_bready : c_bready;
  // AR goes through a register slice: the weight reader's burst logic plus
  // this mux plus the DDR controller's AXI arbiter is too long for one cycle.
  // Ownership only changes while both sides are idle, so a request waiting in
  // the slice always belongs to the current owner.
  logic ar_in_ready, ar_valid;
  logic [AddrW+7:0] ar_data;
  assign ar_valid = sd_busy ? sd_arvalid : c_arvalid;
  assign ar_data  = sd_busy ? {sd_araddr, 8'd0} : {c_araddr, c_arlen};

  skid_buffer #(
      .Width(AddrW + 8)
  ) u_ar_slice (
      .clk_i(clk_i),
      .rst_ni(rst_ni),
      .in_valid_i(ar_valid),
      .in_ready_o(ar_in_ready),
      .in_data_i(ar_data),
      .out_valid_o(m_arvalid_o),
      .out_ready_i(m_arready_i),
      .out_data_o({m_araddr_o, m_arlen_o})
  );

  assign m_rready_o  = sd_busy ? sd_rready : c_rready;
  assign c_awready   = m_awready_i && !sd_busy;
  assign c_wready    = m_wready_i && !sd_busy;
  assign c_bvalid    = m_bvalid_i && !sd_busy;
  assign c_arready   = ar_in_ready && !sd_busy;
  assign c_rvalid    = m_rvalid_i && !sd_busy;
  assign c_rdata     = m_rdata_i;

  sd_ctrl #(
      .ClkHz(ClkHz),
      .SlowHalf(SdSlowHalf),
      .FastHalf(SdFastHalf),
      .PowerClks(SdPowerClks),
      .AddrW(AddrW)
  ) u_sd (
      .clk_i(clk_i),
      .rst_ni(rst_ni),
      .load_i(sd_load),
      .store_i(sd_store),
      .store_addr_i(sd_store_addr_q),
      .store_bytes_i(sd_store_bytes_q),
      .busy_o(sd_busy),
      .ok_o(sd_ok),
      .err_o(sd_err),
      .err_code_o(sd_err_code),
      .img_bytes_o(sd_img_bytes),
      .img_addr_o(sd_img_addr),
      .last_r1_o(sd_last_r1),
      .sd_power_n_o(sd_power_n_o),
      .sd_sclk_o(sd_sclk_o),
      .sd_cs_o(sd_cs_o),
      .sd_mosi_o(sd_mosi_o),
      .sd_miso_i(sd_miso_i),
      .m_awaddr_o(sd_awaddr),
      .m_awvalid_o(sd_awvalid),
      .m_awready_i(m_awready_i && sd_busy),
      .m_wdata_o(sd_wdata),
      .m_wvalid_o(sd_wvalid),
      .m_wready_i(m_wready_i && sd_busy),
      .m_bvalid_i(m_bvalid_i && sd_busy),
      .m_bready_o(sd_bready),
      .m_araddr_o(sd_araddr),
      .m_arvalid_o(sd_arvalid),
      .m_arready_i(ar_in_ready && sd_busy),
      .m_rdata_i(m_rdata_i),
      .m_rvalid_i(m_rvalid_i && sd_busy),
      .m_rready_o(sd_rready)
  );

  // load the card image once the DDR is up
  logic calib_q;
  always_ff @(posedge clk_i, negedge rst_ni) begin
    if (~rst_ni) calib_q <= 1'b0;
    else calib_q <= calib_done_i;
  end

  localparam int ClksPerBit = ClkHz / Baud;

  typedef enum logic [7:0] {
    OP_STATUS = 8'h00,
    OP_WRITE  = 8'h01,
    OP_READ   = 8'h02,
    OP_MATMUL = 8'h03,
    OP_SD_LOAD = 8'h04,
    OP_SD_STORE = 8'h05
  } op_t;

  // ---------------------------------------------------------------------------
  // UART receive. rx_valid / rx_data are registered once more here: the
  // receive buffers below are ~550 flops, too much fanout for one cycle.
  // ---------------------------------------------------------------------------
  logic [7:0] rx_data_raw, rx_data;
  logic rx_valid_raw, rx_valid;

  uart_rx #(
      .ClksPerBit(ClksPerBit)
  ) u_uart_rx (
      .clk_i  (clk_i),
      .rst_ni (rst_ni),
      .rx_i   (uart_rx_i),
      .data_o (rx_data_raw),
      .valid_o(rx_valid_raw)
  );

  always_ff @(posedge clk_i, negedge rst_ni) begin
    if (~rst_ni) rx_valid <= 1'b0;
    else rx_valid <= rx_valid_raw;
  end
  always_ff @(posedge clk_i) rx_data <= rx_data_raw;

  // Received bytes shift in from the top: after 32 bytes the first is at [0].
  // Header fields are read from the top bytes once they are complete.
  logic [31:0][7:0] rx_buf_q;
  always_ff @(posedge clk_i) begin
    if (rx_valid) rx_buf_q <= {rx_data, rx_buf_q[31:1]};
  end

  // ---------------------------------------------------------------------------
  // UART transmit: one byte source at a time, picked by the state machine
  // ---------------------------------------------------------------------------
  logic tx_valid, tx_ready;
  logic [7:0] tx_data;

  uart_tx #(
      .ClksPerBit(ClksPerBit)
  ) u_uart_tx (
      .clk_i  (clk_i),
      .rst_ni (rst_ni),
      .data_i (tx_data),
      .valid_i(tx_valid),
      .ready_o(tx_ready),
      .tx_o   (uart_tx_o)
  );

  // Command state (the state machine is further down)
  typedef enum logic [3:0] {
    S_OP,         // waiting for an opcode
    S_HDR,        // receiving header bytes
    S_STATUS,     // sending the status reply
    S_WR_DATA,    // receiving write data
    S_WR_AXI,     // one beat on AW / W, waiting for B
    S_WR_ACK,     // sending the write ack
    S_RD_ADDR,    // READ: request one beat
    S_RD_DATA,    // READ: waiting for it
    S_RD_SEND,    // READ: sending its 32 bytes
    S_MM_XQ,      // MATMUL: receiving x.q
    S_MM_XS,      // MATMUL: receiving x.s
    S_MM_RUN,     // MATMUL: running, results going out
    S_FLUSH,      // after an abort: wait for DDR reads in flight
    S_SD_ACK      // reply to SD_LOAD / SD_STORE
  } state_t;
  state_t state_q;

  // ---------------------------------------------------------------------------
  // Matmul engine and its weight stream
  // ---------------------------------------------------------------------------
  // Results wait in a FIFO for the UART
  localparam int YFifoDepth = 32;
  logic [31:0] yfifo_data;
  logic yfifo_empty, yfifo_full, yfifo_pop;

  // qmv produces a block of R = 16 results far faster than the UART sends
  // them, and can't be stalled at its output. So a block's first weight beat
  // only goes in when the FIFO has room for all of that block's results:
  // ycredits_q counts FIFO entries not yet promised to a block.
  localparam int R = 16;
  logic [$clog2(YFifoDepth+1)-1:0] ycredits_q;
  logic [15:0] blk_beat_q, beats_per_block_q;  // R * n / P beats per block
  logic block_ok, w_fire;
  assign block_ok = (blk_beat_q != 0) || (ycredits_q >= R);
  assign w_fire = w_valid && w_ready && block_ok;

  fifo #(
      .Width(32),
      .Depth(YFifoDepth)
  ) u_yfifo (
      .clk_i  (clk_i),
      .rst_ni (rst_ni),
      .clear_i(qmv_abort),
      .we_i   (y_valid),
      .wdata_i(y),
      .re_i   (yfifo_pop),
      .rdata_o(yfifo_data),
      .empty_o(yfifo_empty),
      .full_o (yfifo_full)
  );

  logic qmv_abort, qmv_start, qmv_busy, qmv_done;
  logic [15:0] n_q, d_q;
  logic xq_we, xs_we;
  logic [$clog2(MaxN/P)-1:0] xq_addr_q;
  logic [$clog2(MaxN/GS)-1:0] xs_addr_q;
  logic w_valid, w_ready;
  logic [255:0] w_q;
  logic [31:0] w_s;
  logic y_valid;
  logic [31:0] y;

  qmv #(
      .P(P),
      .GS(GS),
      .MaxN(MaxN)
  ) u_qmv (
      .clk_i(clk_i),
      .rst_ni(rst_ni),
      .abort_i(qmv_abort),
      .start_i(qmv_start),
      .n_i(n_q),
      .d_i(d_q),
      .busy_o(qmv_busy),
      .done_o(qmv_done),
      .xq_we_i(xq_we),
      .xq_addr_i(xq_addr_q),
      .xq_data_i(rx_buf_q),
      .xs_we_i(xs_we),
      .xs_addr_i(xs_addr_q),
      .xs_data_i(rx_buf_q[31-:4]),
      .w_valid_i(w_valid && block_ok),
      .w_ready_o(w_ready),
      .w_q_i(w_q),
      .w_s_i(w_s),
      .y_valid_o(y_valid),
      .y_o(y)
  );

  logic rd_start, rd_busy;
  logic [AddrW-1:0] addr_q;
  logic [31:0] rd_beats;
  logic [AddrW-1:0] rd_araddr;
  logic [7:0] rd_arlen;
  logic rd_arvalid, rd_rready;

  wstream_reader #(
      .AddrW(AddrW)
  ) u_reader (
      .clk_i(clk_i),
      .rst_ni(rst_ni),
      .start_i(rd_start),
      .abort_i(qmv_abort),
      .base_i(addr_q),
      .axi_beats_i(rd_beats),
      .busy_o(rd_busy),
      .araddr_o(rd_araddr),
      .arlen_o(rd_arlen),
      .arvalid_o(rd_arvalid),
      .arready_i(c_arready),
      .rdata_i(c_rdata),
      .rvalid_i(c_rvalid && state_q != S_RD_DATA),
      .rready_o(rd_rready),
      .w_valid_o(w_valid),
      .w_ready_i(w_ready && block_ok),
      .w_q_o(w_q),
      .w_s_o(w_s)
  );

  // ---------------------------------------------------------------------------
  // Command state machine
  // ---------------------------------------------------------------------------

  op_t op_q;
  logic [3:0] hdr_len;  // header bytes after the opcode
  logic [5:0] byte_cnt_q;
  logic [31:0] count_q;  // beats / words left
  logic [255:0] beat_q;  // write data or read data for the UART
  logic aw_done_q, w_done_q;
  logic [$clog2(IdleTimeoutClks)-1:0] idle_q;
  logic waiting_for_rx, timeout;
  logic [3:0] jobs_q;
  logic [1:0] ybyte_q;

  // MATMUL's DDR beats, qmv beats = d * n / P, DDR beats = that / 8 * 9.
  // Registered on its own: n and d are stable long before it is used.
  // Two stages (multiply, then shift-add) to fit the 225 MHz clock.
  logic [31:0] mm_axi_beats_q, mm_prod_q;
  always_ff @(posedge clk_i) begin
    mm_prod_q <= 32'(d_q) * 32'(n_q);
    mm_axi_beats_q <= (mm_prod_q / P) / 8 * 9;
  end

  always_comb begin
    case (op_q)
      OP_WRITE, OP_READ, OP_MATMUL, OP_SD_STORE: hdr_len = 4'd8;
      default: hdr_len = 4'd0;
    endcase
  end

  // Only time out while partway through receiving a command
  assign waiting_for_rx = (state_q == S_HDR) || (state_q == S_WR_DATA)
                          || (state_q == S_MM_XQ) || (state_q == S_MM_XS);
  assign timeout = waiting_for_rx && !rx_valid
                   && (idle_q == ($clog2(IdleTimeoutClks))'(IdleTimeoutClks - 1));

  // The header's 8 bytes after the opcode: once complete they are the top 8
  // of rx_buf_q. Word 0 = addr, word 1 = beats, or n | d << 16.
  logic [31:0] hdr_w0, hdr_w1;
  assign hdr_w0 = rx_buf_q[27:24];
  assign hdr_w1 = rx_buf_q[31:28];

  // AXI write: one beat at a time
  assign c_awaddr  = addr_q;
  assign c_awlen   = 8'd0;
  assign c_awvalid = (state_q == S_WR_AXI) && !aw_done_q;
  assign c_wdata   = beat_q;
  assign c_wlast   = 1'b1;
  assign c_wvalid  = (state_q == S_WR_AXI) && !w_done_q;
  assign c_bready  = (state_q == S_WR_AXI);

  // AXI read: single beats for READ, otherwise the weight reader
  always_comb begin
    if (state_q == S_RD_ADDR || state_q == S_RD_DATA) begin
      c_araddr  = addr_q;
      c_arlen   = 8'd0;
      c_arvalid = (state_q == S_RD_ADDR);
      c_rready  = (state_q == S_RD_DATA);
    end else begin
      c_araddr  = rd_araddr;
      c_arlen   = rd_arlen;
      c_arvalid = rd_arvalid;
      c_rready  = rd_rready;
    end
  end

  // UART transmit source
  always_comb begin
    tx_valid  = 1'b0;
    tx_data   = 8'h00;
    yfifo_pop = 1'b0;
    case (state_q)
      S_STATUS: begin
        tx_valid = 1'b1;
        case (byte_cnt_q[3:0])
          4'd0: tx_data = "L";
          4'd1: tx_data = "L";
          4'd2: tx_data = "M";
          4'd3: tx_data = {3'b0, sd_cd_i, sd_err, sd_ok, sd_busy, calib_done_i};
          4'd4: tx_data = {4'b0, sd_err_code};
          4'd5: tx_data = sd_last_r1;
          4'd8: tx_data = sd_img_bytes[7:0];
          4'd9: tx_data = sd_img_bytes[15:8];
          4'd10: tx_data = sd_img_bytes[23:16];
          4'd11: tx_data = sd_img_bytes[31:24];
          default: tx_data = 8'h00;
        endcase
      end
      S_WR_ACK: begin
        tx_valid = 1'b1;
        tx_data  = OP_WRITE;
      end
      S_SD_ACK: begin
        tx_valid = 1'b1;
        tx_data  = op_q;
      end
      S_RD_SEND: begin
        tx_valid = 1'b1;
        tx_data  = beat_q[byte_cnt_q[4:0]*8+:8];
      end
      S_MM_RUN: begin
        tx_valid  = !yfifo_empty;
        tx_data   = yfifo_data[ybyte_q*8+:8];
        yfifo_pop = tx_valid && tx_ready && (ybyte_q == 2'd3);
      end
      default: ;
    endcase
  end

  logic tx_fire;
  assign tx_fire = tx_valid && tx_ready;

  always_ff @(posedge clk_i, negedge rst_ni) begin
    if (~rst_ni) begin
      state_q <= S_OP;
      op_q <= OP_STATUS;
      byte_cnt_q <= '0;
      count_q <= '0;
      addr_q <= '0;
      n_q <= '0;
      d_q <= '0;
      aw_done_q <= 1'b0;
      w_done_q <= 1'b0;
      idle_q <= '0;
      jobs_q <= '0;
      ybyte_q <= '0;
      xq_addr_q <= '0;
      xs_addr_q <= '0;
      xq_we <= 1'b0;
      xs_we <= 1'b0;
      qmv_start <= 1'b0;
      qmv_abort <= 1'b0;
      rd_start <= 1'b0;
      rd_beats <= '0;
      sd_load <= 1'b0;
      sd_store <= 1'b0;
      sd_store_addr_q <= '0;
      sd_store_bytes_q <= '0;
    end else begin
      xq_we <= 1'b0;
      xs_we <= 1'b0;
      qmv_start <= 1'b0;
      qmv_abort <= 1'b0;
      rd_start <= 1'b0;
      sd_load <= calib_done_i && !calib_q;  // boot: load the card image
      sd_store <= 1'b0;

      if (rx_valid || !waiting_for_rx) idle_q <= '0;
      else idle_q <= idle_q + 1;

      // x writes land one cycle after the byte that completes them
      if (xq_we) xq_addr_q <= xq_addr_q + 1;
      if (xs_we) xs_addr_q <= xs_addr_q + 1;

      if (timeout) begin
        state_q <= S_OP;
      end else begin
        case (state_q)
          S_OP: begin
            byte_cnt_q <= '0;
            if (rx_valid) begin
              op_q <= op_t'(rx_data);
              case (rx_data)
                OP_STATUS: state_q <= S_STATUS;
                // the rest only when no SD operation owns the DDR
                OP_WRITE, OP_READ, OP_MATMUL, OP_SD_STORE: if (!sd_busy) state_q <= S_HDR;
                OP_SD_LOAD: begin
                  if (!sd_busy) begin
                    sd_load <= 1'b1;
                    state_q <= S_SD_ACK;
                  end
                end
                default: ;  // unknown opcode: ignore the byte
              endcase
            end
          end

          S_HDR: begin
            if (rx_valid) byte_cnt_q <= byte_cnt_q + 1;
            // the last header byte is in rx_buf_q the cycle after its
            // rx_valid (bytes are many cycles apart, so rx_valid is low then)
            if (!rx_valid && byte_cnt_q == 6'(hdr_len)) begin
              byte_cnt_q <= '0;
              addr_q <= AddrW'(hdr_w0);
              case (op_q)
                OP_WRITE: begin
                  count_q <= hdr_w1;
                  state_q <= (hdr_w1 == 0) ? S_WR_ACK : S_WR_DATA;
                end
                OP_READ: begin
                  count_q <= hdr_w1;
                  state_q <= (hdr_w1 == 0) ? S_OP : S_RD_ADDR;
                end
                OP_SD_STORE: begin
                  sd_store_addr_q <= hdr_w0;
                  sd_store_bytes_q <= hdr_w1;
                  sd_store <= 1'b1;
                  state_q <= S_SD_ACK;
                end
                default: begin  // MATMUL
                  n_q <= hdr_w1[15:0];
                  d_q <= hdr_w1[31:16];
                  count_q <= 32'(hdr_w1[15:0]) / P;
                  xq_addr_q <= '0;
                  xs_addr_q <= '0;
                  state_q <= S_MM_XQ;
                end
              endcase
            end
          end

          S_STATUS: begin
            if (tx_fire) begin
              byte_cnt_q <= byte_cnt_q + 1;
              if (byte_cnt_q == 6'd11) state_q <= S_OP;
            end
          end

          S_WR_DATA: begin
            if (rx_valid) byte_cnt_q <= byte_cnt_q + 1;
            if (!rx_valid && byte_cnt_q == 6'd32) begin
              byte_cnt_q <= '0;
              beat_q <= rx_buf_q;
              aw_done_q <= 1'b0;
              w_done_q <= 1'b0;
              state_q <= S_WR_AXI;
            end
          end

          S_WR_AXI: begin
            if (c_awvalid && c_awready) aw_done_q <= 1'b1;
            if (c_wvalid && c_wready) w_done_q <= 1'b1;
            if (c_bvalid) begin
              addr_q <= addr_q + 32;
              count_q <= count_q - 1;
              state_q <= (count_q == 1) ? S_WR_ACK : S_WR_DATA;
            end
          end

          S_WR_ACK, S_SD_ACK: begin
            if (tx_fire) state_q <= S_OP;
          end

          S_RD_ADDR: begin
            if (c_arready) state_q <= S_RD_DATA;
          end

          S_RD_DATA: begin
            if (c_rvalid) begin
              beat_q <= c_rdata;
              byte_cnt_q <= '0;
              state_q <= S_RD_SEND;
            end
          end

          S_RD_SEND: begin
            if (tx_fire) begin
              byte_cnt_q <= byte_cnt_q + 1;
              if (byte_cnt_q == 6'd31) begin
                addr_q <= addr_q + 32;
                count_q <= count_q - 1;
                state_q <= (count_q == 1) ? S_OP : S_RD_ADDR;
              end
            end
          end

          S_MM_XQ: begin
            if (rx_valid) byte_cnt_q <= byte_cnt_q + 1;
            if (!rx_valid && byte_cnt_q == 6'd32) begin
              byte_cnt_q <= '0;
              xq_we <= 1'b1;
              count_q <= count_q - 1;
              if (count_q == 1) begin
                count_q <= 32'(n_q) / GS;
                state_q <= S_MM_XS;
              end
            end
          end

          S_MM_XS: begin
            if (rx_valid) byte_cnt_q <= byte_cnt_q + 1;
            if (!rx_valid && byte_cnt_q == 6'd4) begin
              byte_cnt_q <= '0;
              xs_we <= 1'b1;
              count_q <= count_q - 1;
              if (count_q == 1) begin
                rd_beats <= mm_axi_beats_q;
                qmv_start <= 1'b1;
                rd_start <= 1'b1;
                ybyte_q <= '0;
                count_q <= 32'(d_q);
                state_q <= S_MM_RUN;
              end
            end
          end

          S_MM_RUN: begin
            if (tx_fire) ybyte_q <= ybyte_q + 1;
            if (yfifo_pop) begin
              count_q <= count_q - 1;
              if (count_q == 1) begin
                jobs_q <= jobs_q + 1;
                state_q <= S_FLUSH;
              end
            end
          end

          S_FLUSH: begin
            if (!rd_busy && !qmv_busy) state_q <= S_OP;
          end

          default: state_q <= S_OP;
        endcase
      end
    end
  end

  always_ff @(posedge clk_i, negedge rst_ni) begin
    if (~rst_ni) begin
      ycredits_q <= '0;
      blk_beat_q <= '0;
      beats_per_block_q <= '0;
    end else if (qmv_start) begin
      ycredits_q <= ($clog2(YFifoDepth+1))'(YFifoDepth);
      blk_beat_q <= '0;
      beats_per_block_q <= 16'(R * 32'(n_q) / P);
    end else begin
      if (w_fire) blk_beat_q <= (blk_beat_q == beats_per_block_q - 1) ? '0 : blk_beat_q + 1;
      ycredits_q <= ycredits_q - ((w_fire && blk_beat_q == 0) ? ($clog2(YFifoDepth+1))'(R) : '0)
                    + ($clog2(YFifoDepth+1))'(yfifo_pop);
    end
  end

  // ---------------------------------------------------------------------------
  // LEDs: 0 heartbeat, 1 DDR calibrated, 2 matmul running,
  //       3 SD: blinks while busy, on when an image loaded / stored fine,
  //       7:4 SD error code after an SD error, else completed matmuls (mod 16)
  // ---------------------------------------------------------------------------
  logic [26:0] heartbeat_q;

  always_ff @(posedge clk_i, negedge rst_ni) begin
    if (~rst_ni) heartbeat_q <= '0;
    else heartbeat_q <= heartbeat_q + 1;
  end

  always_ff @(posedge clk_i) begin
    led_o[0]   <= heartbeat_q[26];
    led_o[1]   <= calib_done_i;
    led_o[2]   <= (state_q == S_MM_RUN);
    led_o[3]   <= sd_busy ? heartbeat_q[23] : sd_ok;
    led_o[7:4] <= sd_err ? sd_err_code : jobs_q;
  end

endmodule

`default_nettype wire
