`default_nettype none

// IEEE 754 binary32 divide a / b, round to nearest even. Pipelined, one
// result per cycle, Latency cycles from valid_i to valid_o; tag_i travels
// alongside.
//
// Long division, one quotient bit per stage: 24 significand bits plus a
// round bit, and the remainder as the sticky bit, which is all correct
// rounding needs.
//
// Simplifications (like fp32_add / fp32_mul):
//   - subnormal inputs are read as zero, subnormal results are flushed to zero
//   - inf / NaN inputs are not handled; overflow saturates to inf
//   - x / 0 is inf, 0 / 0 is the default NaN (0x7FC00000, as RISC-V gives)
module fp32_div #(
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
  localparam int QBits = 25;  // 24 significand bits + the round bit
  localparam int Latency = QBits + 3;

  logic [Latency-1:0] valid_q;
  logic [Latency-1:0][TagW-1:0] tag_q;

  always_ff @(posedge clk_i, negedge rst_ni) begin
    if (~rst_ni) valid_q <= '0;
    else valid_q <= {valid_q[Latency-2:0], valid_i};
  end

  always_ff @(posedge clk_i) tag_q <= {tag_q[Latency-2:0], tag_i};

  assign valid_o = valid_q[Latency-1];
  assign tag_o   = tag_q[Latency-1];

  // S0: unpack. The quotient of the significands is in (1/2, 2): when
  // ma < mb, double the dividend so the first quotient bit is the integer 1.
  typedef struct packed {
    logic sign;
    logic a_zero;
    logic b_zero;
    logic signed [9:0] exp;
  } info_t;

  // stage k's registers: remainder (< 2 * mb), divisor, quotient bits so far
  logic [QBits:0][24:0] rem;
  logic [QBits:0][23:0] mbs;
  logic [QBits:0][QBits-1:0] quo;
  info_t [QBits:0] info;

  always_ff @(posedge clk_i) begin
    logic [23:0] ma, mb;
    ma = {1'b1, a_i[22:0]};
    mb = {1'b1, b_i[22:0]};
    info[0].sign   <= a_i[31] ^ b_i[31];
    info[0].a_zero <= (a_i[30:23] == '0);
    info[0].b_zero <= (b_i[30:23] == '0);
    if (ma >= mb) begin
      rem[0] <= {1'b0, ma};
      info[0].exp <= 10'(a_i[30:23]) - 10'(b_i[30:23]) + 10'sd127;
    end else begin
      rem[0] <= {ma, 1'b0};
      info[0].exp <= 10'(a_i[30:23]) - 10'(b_i[30:23]) + 10'sd126;
    end
    mbs[0] <= mb;
    quo[0] <= '0;
  end

  // S1..S25: one quotient bit each

  for (genvar k = 0; k < QBits; k++) begin : g_stage
    always_ff @(posedge clk_i) begin
      logic ge;
      logic [24:0] r;
      ge = rem[k] >= {1'b0, mbs[k]};
      r = ge ? rem[k] - {1'b0, mbs[k]} : rem[k];
      rem[k+1]  <= {r[23:0], 1'b0};
      mbs[k+1]  <= mbs[k];
      quo[k+1]  <= {quo[k][QBits-2:0], ge};
      info[k+1] <= info[k];
    end
  end

  // S26: round to nearest even (round bit = last quotient bit, sticky =
  // remainder left), S27: pack
  logic [23:0] s26_sig;
  logic s26_carry;
  logic signed [9:0] s26_exp;
  info_t s26_info;

  always_ff @(posedge clk_i) begin
    logic [23:0] sig;
    logic rbit, sticky;
    logic [24:0] rounded;
    sig = quo[QBits][QBits-1:1];
    rbit = quo[QBits][0];
    sticky = (rem[QBits] != '0);
    rounded = {1'b0, sig} + 25'(rbit & (sticky | sig[0]));
    s26_carry <= rounded[24];
    s26_sig   <= rounded[23:0];
    s26_exp   <= info[QBits].exp;
    s26_info  <= info[QBits];
  end

  always_ff @(posedge clk_i) begin
    logic signed [9:0] e;
    e = s26_exp + 10'(s26_carry);
    if (s26_info.a_zero && s26_info.b_zero) y_o <= 32'h7FC0_0000;
    else if (s26_info.b_zero) y_o <= {s26_info.sign, 8'hFF, 23'd0};
    else if (s26_info.a_zero || e <= 0) y_o <= {s26_info.sign, 31'd0};
    else if (e >= 255) y_o <= {s26_info.sign, 8'hFF, 23'd0};
    // on a carry the significand rounded up to 2.0: fraction 0
    else y_o <= {s26_info.sign, e[7:0], s26_carry ? 23'd0 : s26_sig[22:0]};
  end

endmodule

`default_nettype wire
