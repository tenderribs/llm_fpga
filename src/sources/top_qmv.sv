`default_nettype none

// Quantized matrix-vector product (qmv) over UART.
//
// Protocol (8N1, Baud), all little-endian:
//   host -> FPGA:
//     n, d          u16 each: W is d x n
//     x.q           n int8
//     x.s           n/GS float32
//     W             d*n/P beats in qmv stream order (see qmv.sv), each beat
//                   P int8 weights followed by that row's float32 group scale
//   FPGA -> host:   d float32, rows in order, sent as they are computed
//
// If the host stops partway through a job for IdleTimeoutClks, the job is
// aborted and the next byte starts a new one.
//
// LEDs: 0 heartbeat, 1 receiving header / x, 2 streaming W, 3 sending the
//       last results, 7:4 count of completed jobs (mod 16)
module top_qmv #(
    parameter int ClkHz = 200_000_000,
    parameter int Baud = 1_000_000,
    parameter int IdleTimeoutClks = ClkHz / 10,
    parameter int P = 32,
    parameter int GS = 32,
    parameter int R = 16,
    parameter int MaxN = 2048
) (
    input wire logic sysclk_p,
    input wire logic sysclk_n,
    input wire logic cpu_resetn,
    input wire logic uart_rx_i,
    output logic uart_tx_o,
    output logic [7:0] led_o
);

  localparam int ClksPerBit = ClkHz / Baud;
  localparam int BeatBytes = P + 4;

  // ---------------------------------------------------------------------------
  // Clock and reset
  // ---------------------------------------------------------------------------
  logic clk_ibuf, clk;

  IBUFDS u_clk_ibuf (
      .I (sysclk_p),
      .IB(sysclk_n),
      .O (clk_ibuf)
  );

  BUFG u_clk_bufg (
      .I(clk_ibuf),
      .O(clk)
  );

  logic [1:0] rst_sync_q;
  logic rst_n;
  assign rst_n = rst_sync_q[1];

  always_ff @(posedge clk, negedge cpu_resetn) begin
    if (~cpu_resetn) rst_sync_q <= '0;
    else rst_sync_q <= {rst_sync_q[0], 1'b1};
  end

  // ---------------------------------------------------------------------------
  // UART receive
  // ---------------------------------------------------------------------------
  logic [7:0] rx_data;
  logic rx_valid;

  uart_rx #(
      .ClksPerBit(ClksPerBit)
  ) u_uart_rx (
      .clk_i  (clk),
      .rst_ni (rst_n),
      .rx_i   (uart_rx_i),
      .data_o (rx_data),
      .valid_o(rx_valid)
  );

  // Bytes shift in from the top, so after k bytes the first one is at
  // index BeatBytes - k. A full beat has the P weights at [P-1:0] and the
  // scale at [P+3:P]; a header or scale uses the top 4 bytes.
  logic [BeatBytes-1:0][7:0] buf_q;
  logic [P-1:0][7:0] xq_buf_q;

  // ---------------------------------------------------------------------------
  // Engine
  // ---------------------------------------------------------------------------
  logic abort, start, busy, done;
  logic [15:0] n_q, d_q;
  logic xq_we, xs_we;
  logic [$clog2(MaxN/P)-1:0] xq_addr_q;
  logic [$clog2(MaxN/GS)-1:0] xs_addr_q;
  logic w_valid, w_ready;
  logic y_valid;
  logic [31:0] y;

  qmv #(
      .P(P),
      .GS(GS),
      .R(R),
      .MaxN(MaxN)
  ) u_qmv (
      .clk_i(clk),
      .rst_ni(rst_n),
      .abort_i(abort),
      .start_i(start),
      .n_i(n_q),
      .d_i(d_q),
      .busy_o(busy),
      .done_o(done),
      .xq_we_i(xq_we),
      .xq_addr_i(xq_addr_q),
      .xq_data_i(xq_buf_q),
      .xs_we_i(xs_we),
      .xs_addr_i(xs_addr_q),
      .xs_data_i(buf_q[BeatBytes-1-:4]),
      .w_valid_i(w_valid),
      .w_ready_o(w_ready),
      .w_q_i(buf_q[P-1:0]),
      .w_s_i(buf_q[P+3:P]),
      .y_valid_o(y_valid),
      .y_o(y)
  );

  // ---------------------------------------------------------------------------
  // Results: FIFO, then 4 bytes each to the UART. The FIFO only has to hold a
  // burst of R results: the host sends at least GS + 4 bytes per result row,
  // more than the 4 bytes that go back, so the UART TX keeps up on average.
  // ---------------------------------------------------------------------------
  logic [31:0] fifo_rdata;
  logic fifo_empty, fifo_full, fifo_re;
  logic tx_valid, tx_ready;
  logic [1:0] tx_byte_q;
  logic [7:0] tx_data;

  fifo #(
      .Width(32),
      .Depth(2 * R)
  ) u_fifo (
      .clk_i  (clk),
      .rst_ni (rst_n),
      .clear_i(abort),
      .we_i   (y_valid),
      .wdata_i(y),
      .re_i   (fifo_re),
      .rdata_o(fifo_rdata),
      .empty_o(fifo_empty),
      .full_o (fifo_full)
  );

  assign tx_valid = !fifo_empty;
  assign tx_data  = fifo_rdata[tx_byte_q*8+:8];
  assign fifo_re  = tx_valid && tx_ready && (tx_byte_q == 2'd3);

  always_ff @(posedge clk, negedge rst_n) begin
    if (~rst_n) tx_byte_q <= '0;
    else if (abort) tx_byte_q <= '0;
    else if (tx_valid && tx_ready) tx_byte_q <= tx_byte_q + 1;
  end

  uart_tx #(
      .ClksPerBit(ClksPerBit)
  ) u_uart_tx (
      .clk_i  (clk),
      .rst_ni (rst_n),
      .data_i (tx_data),
      .valid_i(tx_valid),
      .ready_o(tx_ready),
      .tx_o   (uart_tx_o)
  );

  // ---------------------------------------------------------------------------
  // Control
  // ---------------------------------------------------------------------------
  typedef enum logic [2:0] {
    HDR,
    XQ,
    XS,
    W,
    DRAIN
  } state_t;
  state_t state_q;

  logic [$clog2(BeatBytes+1)-1:0] byte_cnt_q;  // bytes into the current word
  logic [31:0] word_cnt_q;  // x words / scales / beats so far
  logic [31:0] beats_total_q;
  logic [$clog2(IdleTimeoutClks)-1:0] idle_q;
  logic [3:0] jobs_q;
  logic timeout;

  assign timeout = (state_q != HDR || byte_cnt_q != 0) && !rx_valid
                   && (state_q != DRAIN)
                   && (idle_q == ($clog2(IdleTimeoutClks))'(IdleTimeoutClks - 1));

  always_ff @(posedge clk) begin
    if (rx_valid) buf_q <= {rx_data, buf_q[BeatBytes-1:1]};
    if (rx_valid && state_q == XQ) xq_buf_q <= {rx_data, xq_buf_q[P-1:1]};
  end

  always_ff @(posedge clk, negedge rst_n) begin
    if (~rst_n) begin
      state_q <= HDR;
      byte_cnt_q <= '0;
      word_cnt_q <= '0;
      beats_total_q <= '0;
      idle_q <= '0;
      jobs_q <= '0;
      n_q <= '0;
      d_q <= '0;
      xq_addr_q <= '0;
      xs_addr_q <= '0;
      xq_we <= 1'b0;
      xs_we <= 1'b0;
      start <= 1'b0;
      w_valid <= 1'b0;
      abort <= 1'b0;
    end else begin
      xq_we <= 1'b0;
      xs_we <= 1'b0;
      start <= 1'b0;
      w_valid <= 1'b0;
      abort <= 1'b0;

      if (rx_valid) idle_q <= '0;
      else if (!timeout) idle_q <= idle_q + 1;

      if (timeout) begin
        abort <= 1'b1;
        state_q <= HDR;
        byte_cnt_q <= '0;
        idle_q <= '0;
      end else begin
        case (state_q)
          HDR: begin
            if (rx_valid) begin
              byte_cnt_q <= byte_cnt_q + 1;
              if (byte_cnt_q == 3) begin
                // header bytes 0..2 are the top three of buf_q, byte 3 is
                // rx_data; n = byte1:byte0, d = byte3:byte2
                n_q <= {buf_q[BeatBytes-2], buf_q[BeatBytes-3]};
                d_q <= {rx_data, buf_q[BeatBytes-1]};
                byte_cnt_q <= '0;
                word_cnt_q <= '0;
                xq_addr_q <= '0;
                xs_addr_q <= '0;
                state_q <= XQ;
              end
            end
          end

          XQ: begin
            if (rx_valid) begin
              byte_cnt_q <= byte_cnt_q + 1;
              if (byte_cnt_q == P - 1) begin
                byte_cnt_q <= '0;
                xq_we <= 1'b1;  // xq_buf_q is complete next cycle
                word_cnt_q <= word_cnt_q + 1;
                if (word_cnt_q == 32'(n_q) / P - 1) begin
                  word_cnt_q <= '0;
                  state_q <= XS;
                end
              end
            end
            if (xq_we) xq_addr_q <= xq_addr_q + 1;
          end

          XS: begin
            if (xq_we) xq_addr_q <= xq_addr_q + 1;
            if (xs_we) xs_addr_q <= xs_addr_q + 1;
            if (rx_valid) begin
              byte_cnt_q <= byte_cnt_q + 1;
              if (byte_cnt_q == 3) begin
                byte_cnt_q <= '0;
                xs_we <= 1'b1;  // the scale is in the top 4 bytes next cycle
                word_cnt_q <= word_cnt_q + 1;
                if (word_cnt_q == 32'(n_q) / GS - 1) begin
                  word_cnt_q <= '0;
                  beats_total_q <= 32'(d_q) * 32'(n_q) / P;
                  start <= 1'b1;
                  state_q <= W;
                end
              end
            end
          end

          W: begin
            if (rx_valid) begin
              byte_cnt_q <= byte_cnt_q + 1;
              if (byte_cnt_q == BeatBytes - 1) begin
                byte_cnt_q <= '0;
                // buf_q holds the whole beat next cycle; the engine is ready
                // for every beat of a job, so it is taken that cycle
                w_valid <= 1'b1;
                word_cnt_q <= word_cnt_q + 1;
                if (word_cnt_q == beats_total_q - 1) state_q <= DRAIN;
              end
            end
          end

          // wait for the last results to leave
          DRAIN: begin
            if (!busy && fifo_empty && tx_ready && !y_valid) begin
              jobs_q <= jobs_q + 1;
              state_q <= HDR;
            end
          end

          default: state_q <= HDR;
        endcase
      end
    end
  end

  // ---------------------------------------------------------------------------
  // LEDs
  // ---------------------------------------------------------------------------
  logic [26:0] heartbeat_q;

  always_ff @(posedge clk, negedge rst_n) begin
    if (~rst_n) heartbeat_q <= '0;
    else heartbeat_q <= heartbeat_q + 1;
  end

  always_ff @(posedge clk) begin
    led_o[0]   <= heartbeat_q[26];
    led_o[1]   <= (state_q == XQ) || (state_q == XS) || (state_q == HDR && byte_cnt_q != 0);
    led_o[2]   <= (state_q == W);
    led_o[3]   <= (state_q == DRAIN);
    led_o[7:4] <= jobs_q;
  end

endmodule

`default_nettype wire
