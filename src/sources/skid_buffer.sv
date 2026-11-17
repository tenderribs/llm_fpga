`default_nettype none

// Valid/ready register slice with full throughput: out_valid / out_data come
// straight from flops, and in_ready doesn't depend on out_ready in the same
// cycle, so it cuts combinational paths in both directions. Two entries: the
// output register and a skid register for the beat that arrives while the
// output is stalled.
module skid_buffer #(
    parameter int Width
) (
    input wire logic clk_i,
    input wire logic rst_ni,
    input wire logic in_valid_i,
    output logic in_ready_o,
    input wire logic [Width-1:0] in_data_i,
    output logic out_valid_o,
    input wire logic out_ready_i,
    output logic [Width-1:0] out_data_o
);
  logic [Width-1:0] skid_q;
  logic skid_valid_q;

  assign in_ready_o = !skid_valid_q;

  always_ff @(posedge clk_i, negedge rst_ni) begin
    if (~rst_ni) begin
      out_valid_o <= 1'b0;
      skid_valid_q <= 1'b0;
    end else if (out_ready_i || !out_valid_o) begin
      // output register free this cycle
      if (skid_valid_q) begin
        out_valid_o <= 1'b1;
        skid_valid_q <= 1'b0;
      end else begin
        out_valid_o <= in_valid_i;
      end
    end else if (in_valid_i && in_ready_o) begin
      skid_valid_q <= 1'b1;
    end
  end

  always_ff @(posedge clk_i) begin
    if (out_ready_i || !out_valid_o) out_data_o <= skid_valid_q ? skid_q : in_data_i;
    else if (in_valid_i && in_ready_o) skid_q <= in_data_i;
  end

endmodule

`default_nettype wire
