`default_nettype none

// e^x in binary32, the same operations as fw/model/mathf.c m_expf, each on
// fp32_mul / fp32_add (round to nearest even), so the results are the
// firmware's bits. Pipelined, one result per cycle, Latency cycles; tag_i
// travels alongside.
//
//   x > 88.7228394 -> inf, x < -103.972084 -> 0
//   t = x * log2(e); n = (int)(t < 0 ? t - 0.5 : t + 0.5)
//   r = (x - n * LN2_HI) - n * LN2_LO
//   p = 1 + r (1 + r (1/2 + r (1/6 + r (1/24 + r (1/120 + r (1/720 + r / 5040))))))
//   n > 127: p *= 2, n -= 1; n < -126: p *= 2^-100, n += 100
//   result p * 2^n
//
// Differs from the C function only where the result is subnormal (below
// 2^-126, x < -87.3): the fp32 units flush it to zero. NaN inputs are not
// handled.
module fp32_exp #(
    parameter int TagW = 1
) (
    input wire logic clk_i,
    input wire logic rst_ni,
    input wire logic valid_i,
    input wire logic [TagW-1:0] tag_i,
    input wire logic [31:0] x_i,
    output logic valid_o,
    output logic [TagW-1:0] tag_o,
    output logic [31:0] y_o
);
  localparam logic [31:0] LOG2E = 32'h3FB8AA3B;  // 1.44269502163f
  localparam logic [31:0] LN2_HI = 32'h3F317200;  // 0.693145751953125f
  localparam logic [31:0] LN2_LO = 32'h35BFBE8E;  // 1.42860676533e-6f
  localparam logic [31:0] HALF = 32'h3F000000;
  localparam logic [31:0] BIG = 32'h42B17218;  // 88.7228394f
  localparam logic [31:0] SMALL = 32'h42CFF1B5;  // 103.972084f
  // Horner coefficients, innermost first: 1/5040, then the constants added
  localparam logic [31:0] C5040 = 32'h39500D01;
  localparam logic [31:0] CADD[7] = '{
      32'h3AB60B61,  // 1/720
      32'h3C088889,  // 1/120
      32'h3D2AAAAB,  // 1/24
      32'h3E2AAAAB,  // 1/6
      32'h3F000000,  // 1/2
      32'h3F800000,  // 1
      32'h3F800000  // 1
  };

  // ---------------------------------------------------------------------------
  // E0: range flags
  // ---------------------------------------------------------------------------
  logic e0_valid, e0_over, e0_under;
  logic [31:0] e0_x;
  logic [TagW-1:0] e0_tag;

  always_ff @(posedge clk_i, negedge rst_ni) begin
    if (~rst_ni) e0_valid <= 1'b0;
    else e0_valid <= valid_i;
  end
  always_ff @(posedge clk_i) begin
    e0_x <= x_i;
    e0_tag <= tag_i;
    e0_over <= !x_i[31] && (x_i[30:0] > BIG[30:0]);
    e0_under <= x_i[31] && (x_i[30:0] > SMALL[30:0]);
  end

  typedef struct packed {
    logic [TagW-1:0] tag;
    logic over, under;
  } flags_t;

  // ---------------------------------------------------------------------------
  // t = x * LOG2E; u = t +- 0.5
  // ---------------------------------------------------------------------------
  typedef struct packed {
    flags_t f;
    logic [31:0] x;
  } fx_t;

  logic m1_valid;
  fx_t m1_tag;
  logic [31:0] m1_t;
  fp32_mul #(.TagW($bits(fx_t))) u_m1 (
      .clk_i, .rst_ni, .valid_i(e0_valid), .tag_i({e0_tag, e0_over, e0_under, e0_x}),
      .a_i(e0_x), .b_i(LOG2E), .valid_o(m1_valid), .tag_o(m1_tag), .y_o(m1_t)
  );

  logic t_neg;  // t < 0.0f (-0 is not)
  assign t_neg = m1_t[31] && (m1_t[30:0] != '0);

  logic a1_valid;
  fx_t a1_tag;
  logic [31:0] a1_u;
  fp32_add #(.TagW($bits(fx_t))) u_a1 (
      .clk_i, .rst_ni, .valid_i(m1_valid), .tag_i(m1_tag),
      .a_i(m1_t), .b_i({t_neg, HALF[30:0]}), .valid_o(a1_valid), .tag_o(a1_tag), .y_o(a1_u)
  );

  // ---------------------------------------------------------------------------
  // n = (int)u (toward zero; |u| < 2^8 here), then (float)n
  // ---------------------------------------------------------------------------
  logic c1_valid;
  fx_t c1_tag;
  logic signed [31:0] c1_n;
  always_ff @(posedge clk_i, negedge rst_ni) begin
    if (~rst_ni) c1_valid <= 1'b0;
    else c1_valid <= a1_valid;
  end
  always_ff @(posedge clk_i) begin
    logic [7:0] e;
    logic [31:0] mag;
    e = a1_u[30:23];
    if (e < 8'd127) mag = '0;
    else if (e > 8'd157) mag = 32'h7FFF_FFFF;  // out of range: never here
    else mag = 32'({1'b1, a1_u[22:0]}) << (e - 8'd127) >> 23;
    c1_n <= a1_u[31] ? -$signed(mag) : $signed(mag);
    c1_tag <= a1_tag;
  end

  typedef struct packed {
    fx_t fx;
    logic signed [9:0] n;
  } fxn_t;

  logic i2f_valid;
  fxn_t i2f_tag;
  logic [31:0] i2f_fn;
  int_to_fp32 #(.TagW($bits(fxn_t))) u_i2f (
      .clk_i, .rst_ni, .valid_i(c1_valid), .tag_i({c1_tag, 10'(c1_n)}),
      .a_i(c1_n), .valid_o(i2f_valid), .tag_o(i2f_tag), .y_o(i2f_fn)
  );

  // ---------------------------------------------------------------------------
  // r = (x - fn * LN2_HI) - fn * LN2_LO
  // ---------------------------------------------------------------------------
  logic m2_valid;
  fxn_t m2_tag;
  logic [31:0] m2_hi, m3_lo;
  fp32_mul #(.TagW($bits(fxn_t))) u_m2 (
      .clk_i, .rst_ni, .valid_i(i2f_valid), .tag_i(i2f_tag),
      .a_i(i2f_fn), .b_i(LN2_HI), .valid_o(m2_valid), .tag_o(m2_tag), .y_o(m2_hi)
  );
  fp32_mul #(.TagW(1)) u_m3 (
      .clk_i, .rst_ni, .valid_i(i2f_valid), .tag_i(1'b0),
      .a_i(i2f_fn), .b_i(LN2_LO), .valid_o(), .tag_o(), .y_o(m3_lo)
  );

  typedef struct packed {
    flags_t f;
    logic signed [9:0] n;
    logic [31:0] lo;
  } fnl_t;

  logic a2_valid;
  fnl_t a2_tag;
  logic [31:0] a2_y;
  fp32_add #(.TagW($bits(fnl_t))) u_a2 (
      .clk_i, .rst_ni, .valid_i(m2_valid), .tag_i({m2_tag.fx.f, m2_tag.n, m3_lo}),
      .a_i(m2_tag.fx.x), .b_i({~m2_hi[31], m2_hi[30:0]}), .valid_o(a2_valid), .tag_o(a2_tag),
      .y_o(a2_y)
  );

  typedef struct packed {
    flags_t f;
    logic signed [9:0] n;
  } fn_t;

  logic a3_valid;
  fn_t a3_tag;
  logic [31:0] a3_r;
  fp32_add #(.TagW($bits(fn_t))) u_a3 (
      .clk_i, .rst_ni, .valid_i(a2_valid), .tag_i({a2_tag.f, a2_tag.n}),
      .a_i(a2_y), .b_i({~a2_tag.lo[31], a2_tag.lo[30:0]}), .valid_o(a3_valid), .tag_o(a3_tag),
      .y_o(a3_r)
  );

  // ---------------------------------------------------------------------------
  // Horner: p = 1/5040; 7 times p = c + r * p
  // ---------------------------------------------------------------------------
  typedef struct packed {
    fn_t fn;
    logic [31:0] r;
  } fnr_t;

  logic [7:0] h_valid;
  fnr_t [7:0] h_tag;
  logic [7:0][31:0] h_p;
  assign h_valid[0] = a3_valid;
  assign h_tag[0] = {a3_tag, a3_r};
  assign h_p[0] = C5040;

  for (genvar k = 0; k < 7; k++) begin : g_horner
    logic mv;
    fnr_t mt;
    logic [31:0] my;
    fp32_mul #(.TagW($bits(fnr_t))) u_m (
        .clk_i, .rst_ni, .valid_i(h_valid[k]), .tag_i(h_tag[k]),
        .a_i(h_tag[k].r), .b_i(h_p[k]), .valid_o(mv), .tag_o(mt), .y_o(my)
    );
    fp32_add #(.TagW($bits(fnr_t))) u_a (
        .clk_i, .rst_ni, .valid_i(mv), .tag_i(mt),
        .a_i(CADD[k]), .b_i(my), .valid_o(h_valid[k+1]), .tag_o(h_tag[k+1]), .y_o(h_p[k+1])
    );
  end

  // ---------------------------------------------------------------------------
  // p * 2^n, in two steps where 2^n alone isn't a normal float
  // ---------------------------------------------------------------------------
  logic signed [9:0] n7, n_fin;
  logic [31:0] pre;
  assign n7 = h_tag[7].fn.n;
  always_comb begin
    if (n7 > 10'sd127) begin
      pre = 32'h4000_0000;  // 2
      n_fin = n7 - 10'sd1;
    end else if (n7 < -10'sd126) begin
      pre = 32'h0D80_0000;  // 2^-100
      n_fin = n7 + 10'sd100;
    end else begin
      pre = 32'h3F80_0000;  // 1: p * 1 is p
      n_fin = n7;
    end
  end

  typedef struct packed {
    flags_t f;
    logic [7:0] e;  // n + 127
  } fe_t;

  logic s1_valid;
  fe_t s1_tag;
  logic [31:0] s1_p;
  fp32_mul #(.TagW($bits(fe_t))) u_s1 (
      .clk_i, .rst_ni, .valid_i(h_valid[7]), .tag_i({h_tag[7].fn.f, 8'(n_fin + 10'sd127)}),
      .a_i(h_p[7]), .b_i(pre), .valid_o(s1_valid), .tag_o(s1_tag), .y_o(s1_p)
  );

  logic s2_valid;
  flags_t s2_tag;
  logic [31:0] s2_y;
  fp32_mul #(.TagW($bits(flags_t))) u_s2 (
      .clk_i, .rst_ni, .valid_i(s1_valid), .tag_i(s1_tag.f),
      .a_i(s1_p), .b_i({1'b0, s1_tag.e, 23'd0}), .valid_o(s2_valid), .tag_o(s2_tag), .y_o(s2_y)
  );

  always_ff @(posedge clk_i, negedge rst_ni) begin
    if (~rst_ni) valid_o <= 1'b0;
    else valid_o <= s2_valid;
  end
  always_ff @(posedge clk_i) begin
    tag_o <= s2_tag.tag;
    if (s2_tag.over) y_o <= 32'h7F80_0000;
    else if (s2_tag.under) y_o <= 32'h0000_0000;
    else y_o <= s2_y;
  end

endmodule

`default_nettype wire
