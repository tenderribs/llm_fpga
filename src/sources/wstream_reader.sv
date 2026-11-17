`default_nettype none

// Streams a weight matrix from DDR (AXI4 read) into qmv's weight port.
//
// DDR layout ("qmv DDR format"): the qmv beat stream (see qmv.sv) in chunks
// of 8 beats, each chunk 9 AXI beats of 32 bytes:
//   AXI beat 0:     the 8 beats' group scales, float32, scale j at bytes 4j
//   AXI beats 1..8: the 8 beats' weights, P = 32 int8 each
// so a matrix of B qmv beats (B % 8 == 0) takes B / 8 * 9 AXI beats.
//
// base_i must be 4 KB aligned: bursts are BurstBeats long and must not cross
// a 4 KB boundary.
//
// Reads run ahead of qmv by up to FifoDepth beats; an AR burst is only issued
// when the FIFO is sure to have room for all of it, so R is always ready.
// abort_i stops new bursts; bursts already issued are still received and
// dropped, and busy_o stays high until they are done.
module wstream_reader #(
    parameter int AddrW = 30,
    parameter int BurstBeats = 64,  // 2 KB
    parameter int FifoDepth = 128
) (
    input wire logic clk_i,
    input wire logic rst_ni,

    input wire logic start_i,
    input wire logic abort_i,
    input wire logic [AddrW-1:0] base_i,
    input wire logic [31:0] axi_beats_i,  // AXI beats to read, multiple of 9
    output logic busy_o,

    // AXI4 read address / data
    output logic [AddrW-1:0] araddr_o,
    output logic [7:0] arlen_o,
    output logic arvalid_o,
    input wire logic arready_i,
    input wire logic [255:0] rdata_i,
    input wire logic rvalid_i,
    output logic rready_o,

    // to qmv
    output logic w_valid_o,
    input wire logic w_ready_i,
    output logic [255:0] w_q_o,
    output logic [31:0] w_s_o
);

  if (4096 % (BurstBeats * 32) != 0) begin : g_check_burst
    $error("BurstBeats * 32 must divide 4096");
  end

  localparam int CntW = $clog2(FifoDepth + 1);

  // ---------------------------------------------------------------------------
  // Address side
  // ---------------------------------------------------------------------------
  logic [AddrW-1:0] next_addr_q;
  logic [31:0] to_request_q;  // beats not yet requested
  logic [31:0] to_receive_q;  // beats requested or not, not yet received
  logic [CntW-1:0] credits_q;  // FIFO entries not yet promised to a burst
  logic stopping_q;
  logic [8:0] burst_len;
  logic fifo_pop;

  assign burst_len = (to_request_q < BurstBeats) ? 9'(to_request_q) : 9'(BurstBeats);
  assign araddr_o = next_addr_q;
  assign arlen_o = 8'(burst_len - 1);
  assign arvalid_o = !stopping_q && (to_request_q != 0) && (credits_q >= CntW'(burst_len));
  assign busy_o = (to_receive_q != 0);

  logic ar_fire, r_fire;
  assign ar_fire = arvalid_o && arready_i;
  assign rready_o = 1'b1;
  assign r_fire = rvalid_i;

  always_ff @(posedge clk_i, negedge rst_ni) begin
    if (~rst_ni) begin
      next_addr_q <= '0;
      to_request_q <= '0;
      to_receive_q <= '0;
      credits_q <= CntW'(FifoDepth);
      stopping_q <= 1'b0;
    end else begin
      if (start_i && !busy_o) begin
        next_addr_q <= base_i;
        to_request_q <= axi_beats_i;
        to_receive_q <= axi_beats_i;
        stopping_q <= 1'b0;
      end else begin
        if (abort_i) stopping_q <= 1'b1;
        if (ar_fire) begin
          next_addr_q <= next_addr_q + AddrW'(burst_len) * 32;
          to_request_q <= to_request_q - 32'(burst_len);
        end
        if (r_fire) to_receive_q <= to_receive_q - 1;
        // once stopping, nothing more will be requested
        if (stopping_q && !ar_fire && to_request_q != 0) begin
          to_receive_q <= to_receive_q - to_request_q - 32'(r_fire);
          to_request_q <= '0;
        end
      end

      credits_q <= credits_q - (ar_fire ? CntW'(burst_len) : '0) + CntW'(fifo_pop);
    end
  end

  // ---------------------------------------------------------------------------
  // Data side: FIFO, then split chunks into scales and weight beats
  // ---------------------------------------------------------------------------
  logic [255:0] fifo_data;
  logic fifo_empty, fifo_full;

  fifo #(
      .Width(256),
      .Depth(FifoDepth)
  ) u_fifo (
      .clk_i  (clk_i),
      .rst_ni (rst_ni),
      .clear_i(1'b0),
      .we_i   (r_fire && !stopping_q),
      .wdata_i(rdata_i),
      .re_i   (fifo_pop),
      .rdata_o(fifo_data),
      .empty_o(fifo_empty),
      .full_o (fifo_full)
  );

  logic [3:0] chunk_idx_q;  // 0: scale beat, 1..8: weight beats
  logic [7:0][31:0] scales_q;

  assign w_valid_o = !fifo_empty && (chunk_idx_q != 0) && !stopping_q;
  assign w_q_o = fifo_data;
  assign w_s_o = scales_q[chunk_idx_q-1];
  // the scale beat is consumed here; weight beats when qmv takes them;
  // after an abort, everything is dropped
  assign fifo_pop = !fifo_empty && (stopping_q || chunk_idx_q == 0 || w_ready_i);

  always_ff @(posedge clk_i, negedge rst_ni) begin
    if (~rst_ni) begin
      chunk_idx_q <= '0;
    end else if (start_i && !busy_o) begin
      chunk_idx_q <= '0;
    end else if (fifo_pop && !stopping_q) begin
      chunk_idx_q <= (chunk_idx_q == 4'd8) ? 4'd0 : chunk_idx_q + 1;
    end
  end

  always_ff @(posedge clk_i) begin
    if (fifo_pop && chunk_idx_q == 0) scales_q <= fifo_data;
  end

endmodule

`default_nettype wire
