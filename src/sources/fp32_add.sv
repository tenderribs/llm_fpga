`default_nettype none

// IEEE 754 binary32 add, round to nearest even. Pipelined, one result per
// cycle, Latency cycles from valid_i to valid_o; tag_i travels alongside.
//
// Simplifications (the Q8_0 matmul never produces these values):
//   - subnormal inputs are read as zero, subnormal results are flushed to zero
//   - inf / NaN inputs are not handled; overflow saturates to inf
//
// Significands are carried with 3 extra low bits (guard, round, sticky), which
// is enough for correct round-to-nearest-even.
module fp32_add #(
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
  localparam int Latency = 7;

  logic [Latency-1:0] valid_q;
  logic [Latency-1:0][TagW-1:0] tag_q;

  always_ff @(posedge clk_i, negedge rst_ni) begin
    if (~rst_ni) valid_q <= '0;
    else valid_q <= {valid_q[Latency-2:0], valid_i};
  end

  always_ff @(posedge clk_i) tag_q <= {tag_q[Latency-2:0], tag_i};

  assign valid_o = valid_q[Latency-1];
  assign tag_o   = tag_q[Latency-1];

  // S0: register the inputs (subnormals read as zero) together with the
  // magnitude comparison and both exponent differences, so S1 only selects.
  // With subnormals read as zero, comparing bits 30:0 as an integer compares
  // magnitudes.
  logic [31:0] a_ftz, b_ftz;
  assign a_ftz = (a_i[30:23] == '0) ? {a_i[31], 31'd0} : a_i;
  assign b_ftz = (b_i[30:23] == '0) ? {b_i[31], 31'd0} : b_i;

  logic [31:0] s0_a, s0_b;
  logic s0_a_ge_b;
  logic [7:0] s0_d_ab, s0_d_ba;  // ea - eb, eb - ea (one is used, and >= 0)

  always_ff @(posedge clk_i) begin
    s0_a      <= a_ftz;
    s0_b      <= b_ftz;
    s0_a_ge_b <= a_ftz[30:0] >= b_ftz[30:0];
    s0_d_ab   <= a_ftz[30:23] - b_ftz[30:23];
    s0_d_ba   <= b_ftz[30:23] - a_ftz[30:23];
  end

  // S1: order by magnitude so |hi| >= |lo|
  logic [31:0] hi, lo;
  logic [7:0] exp_diff;
  assign hi = s0_a_ge_b ? s0_a : s0_b;
  assign lo = s0_a_ge_b ? s0_b : s0_a;
  assign exp_diff = s0_a_ge_b ? s0_d_ab : s0_d_ba;

  logic s1_sign, s1_sub, s1_both_neg;
  logic [7:0] s1_exp;
  logic [4:0] s1_shift;  // exponent difference, clamped (see S2)
  logic [23:0] s1_mbig, s1_msmall;

  always_ff @(posedge clk_i) begin
    s1_sign     <= hi[31];
    s1_sub      <= hi[31] ^ lo[31];
    s1_both_neg <= s0_a[31] & s0_b[31];
    s1_exp      <= hi[30:23];
    s1_shift    <= (exp_diff > 8'd31) ? 5'd31 : exp_diff[4:0];
    s1_mbig     <= (hi[30:23] == '0) ? 24'd0 : {1'b1, hi[22:0]};
    s1_msmall   <= (lo[30:23] == '0) ? 24'd0 : {1'b1, lo[22:0]};
  end

  // S2: align the smaller significand. Bits shifted past the round bit fold
  // into sticky. A shift of 31 already moves the hidden bit into sticky, so
  // larger differences can be clamped to 31.
  logic [49:0] s2_shifted;
  assign s2_shifted = {s1_msmall, 26'd0} >> s1_shift;

  logic s2_sign, s2_sub, s2_both_neg;
  logic [7:0] s2_exp;
  logic [26:0] s2_big, s2_small;  // significand, guard, round, sticky

  always_ff @(posedge clk_i) begin
    s2_sign     <= s1_sign;
    s2_sub      <= s1_sub;
    s2_both_neg <= s1_both_neg;
    s2_exp      <= s1_exp;
    s2_big      <= {s1_mbig, 3'd0};
    s2_small    <= {s2_shifted[49:24], |s2_shifted[23:0]};
  end

  // S3: add or subtract magnitudes
  logic s3_sign, s3_both_neg;
  logic [7:0] s3_exp;
  logic [27:0] s3_sum;

  always_ff @(posedge clk_i) begin
    s3_sign     <= s2_sign;
    s3_both_neg <= s2_both_neg;
    s3_exp      <= s2_exp;
    s3_sum      <= s2_sub ? 28'(s2_big - s2_small) : 28'(s2_big) + 28'(s2_small);
  end

  // S4: leading zero count for the subtract case
  logic [4:0] lzc;
  always_comb begin
    lzc = 5'd27;
    for (int i = 0; i <= 26; i++) begin
      if (s3_sum[i]) lzc = 5'(26 - i);
    end
  end

  logic s4_sign, s4_both_neg, s4_carry, s4_zero;
  logic [7:0] s4_exp;
  logic [4:0] s4_lzc;
  logic [27:0] s4_sum;

  always_ff @(posedge clk_i) begin
    s4_sign     <= s3_sign;
    s4_both_neg <= s3_both_neg;
    s4_exp      <= s3_exp;
    s4_sum      <= s3_sum;
    s4_carry    <= s3_sum[27];
    s4_zero     <= (s3_sum == '0);
    s4_lzc      <= lzc;
  end

  // S5: normalize so the hidden bit is at bit 26
  logic s5_sign, s5_zero, s5_both_neg;
  logic signed [9:0] s5_exp;
  logic [26:0] s5_m;

  always_ff @(posedge clk_i) begin
    s5_sign     <= s4_sign;
    s5_zero     <= s4_zero;
    s5_both_neg <= s4_both_neg;
    if (s4_carry) begin
      s5_exp <= 10'(s4_exp) + 10'sd1;
      s5_m   <= {s4_sum[27:2], s4_sum[1] | s4_sum[0]};
    end else begin
      s5_exp <= 10'(s4_exp) - 10'(s4_lzc);
      s5_m   <= s4_sum[26:0] << s4_lzc;
    end
  end

  // S6: round to nearest even, pack
  logic [24:0] s6_rounded;
  logic signed [9:0] s6_exp;
  always_comb begin
    s6_rounded = {1'b0, s5_m[26:3]} + 25'(s5_m[2] & (s5_m[1] | s5_m[0] | s5_m[3]));
    s6_exp = s5_exp + 10'(s6_rounded[24]);
  end

  logic [31:0] y_q;
  assign y_o = y_q;

  always_ff @(posedge clk_i) begin
    // exact zero: x + (-x) = +0, (-0) + (-0) = -0
    if (s5_zero) y_q <= {s5_both_neg, 31'd0};
    else if (s6_exp <= 0) y_q <= {s5_sign, 31'd0};
    else if (s6_exp >= 255) y_q <= {s5_sign, 8'hFF, 23'd0};
    // on a carry out of rounding the significand is 1.0, fraction bits 0
    else y_q <= {s5_sign, s6_exp[7:0], s6_rounded[24] ? 23'd0 : s6_rounded[22:0]};
  end

endmodule

`default_nettype wire
