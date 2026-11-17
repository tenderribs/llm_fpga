`default_nettype none

// Quantized matrix-vector engine: y = W (d x n) @ x (n), with W and x in
// llama2.c's Q8_0 format (int8 values, one float32 scale per group of GS
// consecutive values along n). Bit-exact with runq.c matmul():
//
//   for each row:  val = 0.0f
//                  for each group g:  ival = sum of GS int8 * int8 products
//                                     val  = val + ((float)ival * w_s) * x_s[g]
//
// Usage:
//   1. while idle, write x: xq (P bytes per word) and xs (one scale per group)
//   2. pulse start_i with n_i, d_i
//   3. stream W in, P bytes per beat with that beat's group scale
//   4. y comes out one float per cycle with y_valid_o, rows in order; the
//      consumer must always accept it. done_o pulses after the last row.
//
// Weight stream order: the float accumulation of one row is a chain of
// dependent adds, so the engine works on R rows round-robin to hide the adder
// pipeline. The host lays W out as
//   for each block of R rows:  for each group g:  for each row r in the block:
//     for each of the GS/P beats k of the group:  W[row, g*GS + k*P +: P]
// w_s_i must hold that row's scale for group g on every beat of the group.
//
// Requirements: n % GS == 0, GS % P == 0, d % R == 0, n <= MaxN.
module qmv #(
    parameter int P = 32,      // int8 lanes: weight bytes per beat
    parameter int GS = 32,     // quantization group size
    parameter int R = 16,      // rows interleaved; must exceed the add latency
    parameter int MaxN = 2048
) (
    input wire logic clk_i,
    input wire logic rst_ni,
    input wire logic abort_i,  // drop the current job and go idle

    input wire logic start_i,
    input wire logic [15:0] n_i,
    input wire logic [15:0] d_i,
    output logic busy_o,
    output logic done_o,

    // x load port, used while idle
    input wire logic xq_we_i,
    input wire logic [$clog2(MaxN/P)-1:0] xq_addr_i,
    input wire logic [P*8-1:0] xq_data_i,  // byte i = x.q[addr*P + i]
    input wire logic xs_we_i,
    input wire logic [$clog2(MaxN/GS)-1:0] xs_addr_i,
    input wire logic [31:0] xs_data_i,

    // weight stream
    input wire logic w_valid_i,
    output logic w_ready_o,
    input wire logic [P*8-1:0] w_q_i,  // byte i = lane i
    input wire logic [31:0] w_s_i,

    // results
    output logic y_valid_o,
    output logic [31:0] y_o
);

  localparam int BeatsPerGroup = GS / P;
  localparam int XqDepth = MaxN / P;
  localparam int XsDepth = MaxN / GS;
  localparam int AddLatency = 7;  // fp32_add Latency

  // The row's previous sum is read when a new term enters the adder and
  // written AddLatency cycles later. The same row comes back R*BeatsPerGroup
  // beats later at the earliest, so that has to be longer.
  if (R * BeatsPerGroup <= AddLatency) begin : g_check_r
    $error("R * GS / P must exceed the fp32_add latency");
  end
  if (GS % P != 0) begin : g_check_gs
    $error("GS must be a multiple of P");
  end

  // ---------------------------------------------------------------------------
  // x buffers
  // ---------------------------------------------------------------------------
  logic [P*8-1:0] xq_mem[XqDepth];
  logic [31:0] xs_mem[XsDepth];

  logic [$clog2(XqDepth)-1:0] xq_raddr;
  logic [$clog2(XsDepth)-1:0] xs_raddr;
  logic [P*8-1:0] xq_rdata;
  logic [31:0] xs_rdata;

  always_ff @(posedge clk_i) begin
    if (xq_we_i) xq_mem[xq_addr_i] <= xq_data_i;
    if (xs_we_i) xs_mem[xs_addr_i] <= xs_data_i;
    xq_rdata <= xq_mem[xq_raddr];
    xs_rdata <= xs_mem[xs_raddr];
  end

  // ---------------------------------------------------------------------------
  // Beat counters: which (block, group, row, beat-in-group) the next beat is
  // ---------------------------------------------------------------------------
  logic [15:0] groups_q, blocks_q;  // per job: n / GS, d / R
  logic [15:0] g_q, blk_q;
  logic [$clog2(R)-1:0] r_q;
  logic [$clog2(BeatsPerGroup+1)-1:0] k_q;
  logic in_busy_q;  // still expecting weight beats
  logic busy_q;     // from start until the last row is out
  logic fire;

  assign w_ready_o = in_busy_q;
  assign fire = w_valid_i & w_ready_o;

  logic last_k, last_r, last_g, last_blk;
  assign last_k   = (k_q == BeatsPerGroup - 1);
  assign last_r   = (r_q == R - 1);
  assign last_g   = (g_q == groups_q - 1);
  assign last_blk = (blk_q == blocks_q - 1);

  assign xq_raddr = ($clog2(XqDepth))'(g_q * BeatsPerGroup + 16'(k_q));
  assign xs_raddr = ($clog2(XsDepth))'(g_q);

  always_ff @(posedge clk_i, negedge rst_ni) begin
    if (~rst_ni) begin
      in_busy_q <= 1'b0;
      groups_q <= '0;
      blocks_q <= '0;
      g_q <= '0;
      blk_q <= '0;
      r_q <= '0;
      k_q <= '0;
    end else if (abort_i) begin
      in_busy_q <= 1'b0;
    end else if (start_i && !busy_o) begin
      in_busy_q <= 1'b1;
      groups_q <= n_i / 16'(GS);
      blocks_q <= d_i / 16'(R);
      g_q <= '0;
      blk_q <= '0;
      r_q <= '0;
      k_q <= '0;
    end else if (fire) begin
      k_q <= last_k ? '0 : k_q + 1;
      if (last_k) begin
        r_q <= last_r ? '0 : r_q + 1;
        if (last_r) begin
          g_q <= last_g ? '0 : g_q + 1;
          if (last_g) begin
            blk_q <= blk_q + 1;
            if (last_blk) in_busy_q <= 1'b0;
          end
        end
      end
    end
  end

  // ---------------------------------------------------------------------------
  // Integer dot product, one beat per cycle
  // ---------------------------------------------------------------------------
  // Per-beat metadata that travels down the pipeline
  typedef struct packed {
    logic [$clog2(R)-1:0] r;
    logic first_g;
    logic last_g;
    logic first_k;
    logic last_k;
  } meta_t;

  // c1: beat registered, x read data arrives
  logic c1_valid;
  meta_t c1_meta;
  logic [P*8-1:0] c1_w;
  logic [31:0] c1_ws;

  always_ff @(posedge clk_i, negedge rst_ni) begin
    if (~rst_ni) c1_valid <= 1'b0;
    else c1_valid <= fire & ~abort_i;
  end

  always_ff @(posedge clk_i) begin
    c1_meta <= '{r: r_q, first_g: (g_q == 0), last_g: last_g, first_k: (k_q == 0), last_k: last_k};
    c1_w <= w_q_i;
    c1_ws <= w_s_i;
  end

  // c2: P products
  logic c2_valid;
  meta_t c2_meta;
  logic [31:0] c2_ws, c2_xs;
  logic signed [15:0] c2_prod[P];

  always_ff @(posedge clk_i, negedge rst_ni) begin
    if (~rst_ni) c2_valid <= 1'b0;
    else c2_valid <= c1_valid & ~abort_i;
  end

  always_ff @(posedge clk_i) begin
    c2_meta <= c1_meta;
    c2_ws <= c1_ws;
    c2_xs <= xs_rdata;
    for (int i = 0; i < P; i++) begin
      c2_prod[i] <= $signed(c1_w[i*8+:8]) * $signed(xq_rdata[i*8+:8]);
    end
  end

  // c3..: adder tree, registered after every two levels. Each stage owns its
  // registers; stage s reads stage s-1 (or c2 for s = 0).
  localparam int TreeLevels = $clog2(P);
  localparam int TreeStages = (TreeLevels + 1) / 2;

  for (genvar s = 0; s < TreeStages; s++) begin : g_tree
    logic signed [31:0] in[P];
    logic in_valid;
    meta_t in_meta;
    logic [31:0] in_ws, in_xs;

    if (s == 0) begin : g_first
      always_comb begin
        for (int i = 0; i < P; i++) in[i] = 32'(c2_prod[i]);
      end
      assign in_valid = c2_valid;
      assign in_meta  = c2_meta;
      assign in_ws    = c2_ws;
      assign in_xs    = c2_xs;
    end else begin : g_next
      assign in       = g_tree[s-1].q;
      assign in_valid = g_tree[s-1].q_valid;
      assign in_meta  = g_tree[s-1].q_meta;
      assign in_ws    = g_tree[s-1].q_ws;
      assign in_xs    = g_tree[s-1].q_xs;
    end

    // two levels of pairwise sums (one on the last stage if levels are odd)
    logic signed [31:0] lvl1[P];
    logic signed [31:0] lvl2[P];
    always_comb begin
      for (int i = 0; i < P; i++) begin
        lvl1[i] = '0;
        lvl2[i] = '0;
      end
      for (int i = 0; i < P / 2; i++) lvl1[i] = in[2*i] + in[2*i+1];
      if (2 * s + 1 < TreeLevels) begin
        for (int i = 0; i < P / 4; i++) lvl2[i] = lvl1[2*i] + lvl1[2*i+1];
      end else begin
        for (int i = 0; i < P / 2; i++) lvl2[i] = lvl1[i];
      end
    end

    logic signed [31:0] q[P];
    logic q_valid;
    meta_t q_meta;
    logic [31:0] q_ws, q_xs;

    always_ff @(posedge clk_i, negedge rst_ni) begin
      if (~rst_ni) q_valid <= 1'b0;
      else q_valid <= in_valid & ~abort_i;
    end

    always_ff @(posedge clk_i) begin
      q <= lvl2;
      q_meta <= in_meta;
      q_ws <= in_ws;
      q_xs <= in_xs;
    end
  end

  logic signed [31:0] tree_sum;
  logic tree_valid;
  meta_t tree_meta;
  logic [31:0] tree_ws, tree_xs;
  assign tree_sum   = g_tree[TreeStages-1].q[0];
  assign tree_valid = g_tree[TreeStages-1].q_valid;
  assign tree_meta  = g_tree[TreeStages-1].q_meta;
  assign tree_ws    = g_tree[TreeStages-1].q_ws;
  assign tree_xs    = g_tree[TreeStages-1].q_xs;

  // Group sum: add up the GS/P beats of one row's group (consecutive beats)
  logic gsum_valid;
  meta_t gsum_meta;
  logic [31:0] gsum_ws, gsum_xs;
  logic signed [31:0] gsum, gsum_acc_q;

  assign gsum = tree_meta.first_k ? tree_sum : gsum_acc_q + tree_sum;

  always_ff @(posedge clk_i, negedge rst_ni) begin
    if (~rst_ni) gsum_valid <= 1'b0;
    else gsum_valid <= tree_valid & tree_meta.last_k & ~abort_i;
  end

  always_ff @(posedge clk_i) begin
    if (tree_valid) gsum_acc_q <= gsum;
    gsum_meta <= tree_meta;
    gsum_ws <= tree_ws;
    gsum_xs <= tree_xs;
  end

  // ---------------------------------------------------------------------------
  // Float: ((float)ival * w_s) * x_s, then accumulate per row
  // ---------------------------------------------------------------------------
  typedef struct packed {
    logic [$clog2(R)-1:0] r;
    logic first_g;
    logic last_g;
  } fmeta_t;

  fmeta_t gsum_fmeta;
  assign gsum_fmeta = '{r: gsum_meta.r, first_g: gsum_meta.first_g, last_g: gsum_meta.last_g};

  logic f1_valid, m1_valid, m2_valid, acc_valid;
  logic [31:0] f1_y, m1_y, m2_y, acc_y;
  logic [$bits(fmeta_t)+64-1:0] f1_tag;
  logic [$bits(fmeta_t)+32-1:0] m1_tag;
  fmeta_t m2_meta, acc_meta;

  int_to_fp32 #(
      .TagW($bits(fmeta_t) + 64)
  ) u_i2f (
      .clk_i  (clk_i),
      .rst_ni (rst_ni),
      .valid_i(gsum_valid),
      .tag_i  ({gsum_fmeta, gsum_ws, gsum_xs}),
      .a_i    (gsum_acc_q),
      .valid_o(f1_valid),
      .tag_o  (f1_tag),
      .y_o    (f1_y)
  );

  fp32_mul #(
      .TagW($bits(fmeta_t) + 32)
  ) u_mul_ws (
      .clk_i  (clk_i),
      .rst_ni (rst_ni),
      .valid_i(f1_valid),
      .tag_i  ({f1_tag[$bits(f1_tag)-1:64], f1_tag[31:0]}),  // meta, x_s
      .a_i    (f1_y),
      .b_i    (f1_tag[63:32]),  // w_s
      .valid_o(m1_valid),
      .tag_o  (m1_tag),
      .y_o    (m1_y)
  );

  fp32_mul #(
      .TagW($bits(fmeta_t))
  ) u_mul_xs (
      .clk_i  (clk_i),
      .rst_ni (rst_ni),
      .valid_i(m1_valid),
      .tag_i  (m1_tag[$bits(m1_tag)-1:32]),
      .a_i    (m1_y),
      .b_i    (m1_tag[31:0]),  // x_s
      .valid_o(m2_valid),
      .tag_o  (m2_meta),
      .y_o    (m2_y)
  );

  // Per-row running sums. The first group adds to +0.0, as runq.c's
  // val = 0.0f does (this matters for the sign of a zero result).
  logic [31:0] row_sum[R];

  fp32_add #(
      .TagW($bits(fmeta_t))
  ) u_acc (
      .clk_i  (clk_i),
      .rst_ni (rst_ni),
      .valid_i(m2_valid),
      .tag_i  (m2_meta),
      .a_i    (m2_meta.first_g ? 32'd0 : row_sum[m2_meta.r]),
      .b_i    (m2_y),
      .valid_o(acc_valid),
      .tag_o  (acc_meta),
      .y_o    (acc_y)
  );

  always_ff @(posedge clk_i) begin
    if (acc_valid) row_sum[acc_meta.r] <= acc_y;
  end

  // C's order is val + term; IEEE addition is commutative, so a/b order
  // doesn't matter. Gated with busy so results still in flight after an
  // abort don't come out.
  always_ff @(posedge clk_i, negedge rst_ni) begin
    if (~rst_ni) y_valid_o <= 1'b0;
    else y_valid_o <= acc_valid & acc_meta.last_g & busy_q & ~abort_i;
  end

  always_ff @(posedge clk_i) y_o <= acc_y;

  // ---------------------------------------------------------------------------
  // Job tracking: busy from start until the last row has come out
  // ---------------------------------------------------------------------------
  logic [15:0] rows_left_q;
  assign busy_o = busy_q;

  always_ff @(posedge clk_i, negedge rst_ni) begin
    if (~rst_ni) begin
      busy_q <= 1'b0;
      rows_left_q <= '0;
      done_o <= 1'b0;
    end else begin
      done_o <= 1'b0;
      if (abort_i) begin
        busy_q <= 1'b0;
      end else if (start_i && !busy_q) begin
        busy_q <= 1'b1;
        rows_left_q <= d_i;
      end else if (busy_q && y_valid_o) begin
        rows_left_q <= rows_left_q - 1;
        if (rows_left_q == 1) begin
          busy_q <= 1'b0;
          done_o <= 1'b1;
        end
      end
    end
  end

endmodule

`default_nettype wire
