`default_nettype none

// Signed int32 to IEEE 754 binary32, round to nearest even (as C's
// (float) cast). Pipelined, Latency cycles; tag_i travels alongside.
module int_to_fp32 #(
    parameter int TagW = 1
) (
    input wire logic clk_i,
    input wire logic rst_ni,
    input wire logic valid_i,
    input wire logic [TagW-1:0] tag_i,
    input wire logic signed [31:0] a_i,
    output logic valid_o,
    output logic [TagW-1:0] tag_o,
    output logic [31:0] y_o
);
  localparam int Latency = 3;

  logic [Latency-1:0] valid_q;
  logic [Latency-1:0][TagW-1:0] tag_q;

  always_ff @(posedge clk_i, negedge rst_ni) begin
    if (~rst_ni) valid_q <= '0;
    else valid_q <= {valid_q[Latency-2:0], valid_i};
  end

  always_ff @(posedge clk_i) tag_q <= {tag_q[Latency-2:0], tag_i};

  assign valid_o = valid_q[Latency-1];
  assign tag_o   = tag_q[Latency-1];

  // S1: sign and magnitude (|-2^31| = 2^31 fits in 32 unsigned bits)
  logic s1_sign;
  logic [31:0] s1_mag;

  always_ff @(posedge clk_i) begin
    s1_sign <= a_i[31];
    s1_mag  <= a_i[31] ? 32'(-a_i) : 32'(a_i);
  end

  // S2: normalize so the leading one is at bit 31
  logic [4:0] lzc;
  always_comb begin
    lzc = 5'd0;
    for (int i = 0; i <= 31; i++) begin
      if (s1_mag[i]) lzc = 5'(31 - i);
    end
  end

  logic s2_sign, s2_zero;
  logic [4:0] s2_lzc;
  logic [31:0] s2_norm;

  always_ff @(posedge clk_i) begin
    s2_sign <= s1_sign;
    s2_zero <= (s1_mag == '0);
    s2_lzc  <= lzc;
    s2_norm <= s1_mag << lzc;
  end

  // S3: round to nearest even, pack. Exponent = 127 + 31 - lzc.
  logic [23:0] rounded;
  logic [7:0] exp;
  always_comb begin
    rounded = {1'b0, s2_norm[30:8]} + 24'(s2_norm[7] & ((|s2_norm[6:0]) | s2_norm[8]));
    exp = 8'(158 - 32'(s2_lzc)) + 8'(rounded[23]);
  end

  always_ff @(posedge clk_i) begin
    if (s2_zero) y_o <= 32'd0;
    else y_o <= {s2_sign, exp, rounded[22:0]};
  end

endmodule

`default_nettype wire
