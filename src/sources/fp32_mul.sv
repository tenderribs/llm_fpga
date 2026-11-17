`default_nettype none

// IEEE 754 binary32 multiply, round to nearest even. Pipelined, one result per
// cycle, Latency cycles from valid_i to valid_o; tag_i travels alongside.
//
// Simplifications (the Q8_0 matmul never produces these values):
//   - subnormal inputs are read as zero, subnormal results are flushed to zero
//   - inf / NaN inputs are not handled; overflow saturates to inf
module fp32_mul #(
    parameter int TagW = 1
) (
    input wire logic clk_i,
    input wire logic rst_ni,
    input wire logic valid_i,
    input wire logic [TagW-1:0] tag_i,
    input wire logic [31:0] a_i,
    input wire logic [31:0] b_i,
    output logic valid_o,
    output logic [TagW-1:0] tag_o,
    output logic [31:0] y_o
);
  localparam int Latency = 5;

  // valid and tag ride along a shift register
  logic [Latency-1:0] valid_q;
  logic [Latency-1:0][TagW-1:0] tag_q;

  always_ff @(posedge clk_i, negedge rst_ni) begin
    if (~rst_ni) valid_q <= '0;
    else valid_q <= {valid_q[Latency-2:0], valid_i};
  end

  always_ff @(posedge clk_i) tag_q <= {tag_q[Latency-2:0], tag_i};

  assign valid_o = valid_q[Latency-1];
  assign tag_o   = tag_q[Latency-1];

  // S1: unpack
  logic s1_sign, s1_zero;
  logic signed [9:0] s1_exp;  // biased sum of exponents, minus one bias
  logic [23:0] s1_ma, s1_mb;

  always_ff @(posedge clk_i) begin
    s1_sign <= a_i[31] ^ b_i[31];
    s1_zero <= (a_i[30:23] == '0) || (b_i[30:23] == '0);
    s1_exp  <= 10'(a_i[30:23]) + 10'(b_i[30:23]) - 10'sd127;
    s1_ma   <= {1'b1, a_i[22:0]};
    s1_mb   <= {1'b1, b_i[22:0]};
  end

  // S2, S3: 24x24 significand product (two registers so it maps onto DSP48
  // M and P registers)
  logic s2_sign, s2_zero, s3_sign, s3_zero;
  logic signed [9:0] s2_exp, s3_exp;
  logic [47:0] s2_p, s3_p;

  always_ff @(posedge clk_i) begin
    s2_sign <= s1_sign;
    s2_zero <= s1_zero;
    s2_exp  <= s1_exp;
    s2_p    <= s1_ma * s1_mb;
    s3_sign <= s2_sign;
    s3_zero <= s2_zero;
    s3_exp  <= s2_exp;
    s3_p    <= s2_p;
  end

  // S4: normalize. The product of two [1, 2) significands is in [1, 4).
  logic s4_sign, s4_zero, s4_guard, s4_sticky;
  logic signed [9:0] s4_exp;
  logic [22:0] s4_frac;

  always_ff @(posedge clk_i) begin
    s4_sign <= s3_sign;
    s4_zero <= s3_zero;
    if (s3_p[47]) begin
      s4_exp    <= s3_exp + 10'sd1;
      s4_frac   <= s3_p[46:24];
      s4_guard  <= s3_p[23];
      s4_sticky <= |s3_p[22:0];
    end else begin
      s4_exp    <= s3_exp;
      s4_frac   <= s3_p[45:23];
      s4_guard  <= s3_p[22];
      s4_sticky <= |s3_p[21:0];
    end
  end

  // S5: round to nearest even, pack
  logic [23:0] s5_rounded;
  logic signed [9:0] s5_exp;
  always_comb begin
    s5_rounded = {1'b0, s4_frac} + 24'(s4_guard & (s4_sticky | s4_frac[0]));
    // carry out of the fraction: significand rounded up to 2.0
    s5_exp = s4_exp + 10'(s5_rounded[23]);
  end

  always_ff @(posedge clk_i) begin
    if (s4_zero || s5_exp <= 0) y_o <= {s4_sign, 31'd0};
    else if (s5_exp >= 255) y_o <= {s4_sign, 8'hFF, 23'd0};
    else y_o <= {s4_sign, s5_exp[7:0], s5_rounded[22:0]};
  end

endmodule

`default_nettype wire
