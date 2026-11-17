`default_nettype none

// Synchronous FIFO, first-word fall-through: rdata_o is the head entry
// whenever empty_o is low. Depth must be a power of two.
module fifo #(
    parameter int Width,
    parameter int Depth
) (
    input wire logic clk_i,
    input wire logic rst_ni,
    input wire logic clear_i,
    input wire logic we_i,
    input wire logic [Width-1:0] wdata_i,
    input wire logic re_i,
    output logic [Width-1:0] rdata_o,
    output logic empty_o,
    output logic full_o
);
  localparam int AW = $clog2(Depth);

  logic [Width-1:0] mem[Depth];
  logic [AW:0] wptr_q, rptr_q;  // one extra bit tells full from empty

  assign empty_o = (wptr_q == rptr_q);
  assign full_o  = (wptr_q[AW-1:0] == rptr_q[AW-1:0]) && (wptr_q[AW] != rptr_q[AW]);
  assign rdata_o = mem[rptr_q[AW-1:0]];

  always_ff @(posedge clk_i) begin
    if (we_i && !full_o) mem[wptr_q[AW-1:0]] <= wdata_i;
  end

  always_ff @(posedge clk_i, negedge rst_ni) begin
    if (~rst_ni) begin
      wptr_q <= '0;
      rptr_q <= '0;
    end else if (clear_i) begin
      wptr_q <= '0;
      rptr_q <= '0;
    end else begin
      if (we_i && !full_o) wptr_q <= wptr_q + 1;
      if (re_i && !empty_o) rptr_q <= rptr_q + 1;
    end
  end

endmodule

`default_nettype wire
