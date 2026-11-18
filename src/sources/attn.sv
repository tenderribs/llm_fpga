`default_nettype none

// Attention for one head, the two float loops of runq.c's forward(), with
// the same float operations in the same order (so the results are the same
// bits as on a CPU without FMA):
//
//   mode 0, scores:   y[t] = sum over i of q[i] * k_t[i]      (t < nt)
//   mode 1, weighted: y[i] = sum over t of a[t] * v_t[i]      (i < hs)
//
// each sum starting from +0 and adding its terms in index order. The CPU and
// the vector unit do the rest (scaling, softmax).
//
// The A RAM holds q (mode 0) or the attention weights a (mode 1). K or V
// streams in as 32-byte beats (8 floats), from DDR, one beat per cycle into
// 8 lanes (8 multiply-adds per cycle):
//   mode 0: K in blocks of Tb = 128 timesteps, each block [i][t % 128]: a
//           beat is 8 timesteps' element i, lane j of beat c (0..15) of a row
//           is timestep c * 8 + j; ceil(nt / 128) * hs * 16 beats
//   mode 1: V as rows [t][i]: lane j of beat c is element c * 8 + j;
//           nt * hs / 8 beats
// Each sum is a chain of dependent adds: lane j's sum number c gets its
// next term c beats later in the next row (16 beats in mode 0, hs / 8 in
// mode 1). A beat waits when its sums were used in the last 7 beats (the
// multiply-add loop's latency), which only happens in mode 1 with hs < 64.
// hs must be a multiple of 8.
//
// Results come out on y_* (index, value), one per cycle, through a FIFO:
// mode 0 gives ceil(nt / 128) * 128 scores (the padding timesteps' are
// garbage), mode 1 hs sums.
module attn #(
    parameter int MaxA = 4096,  // A RAM words: max head size / timesteps
    parameter int Lanes = 8
) (
    input wire logic clk_i,
    input wire logic rst_ni,

    input wire logic start_i,
    input wire logic mode_i,  // 0 scores, 1 weighted sum
    input wire logic [8:0] hs_i,
    input wire logic [15:0] nt_i,
    input wire logic [31:0] beats_i,  // mode 0: ceil(nt / 128) * hs * 16, mode 1: nt * hs / 8
    output logic busy_o,
    output logic done_o,  // one cycle, after the last result

    // A RAM (q or a)
    input wire logic a_we_i,
    input wire logic [$clog2(MaxA)-1:0] a_addr_i,
    input wire logic [31:0] a_data_i,

    // K / V beats
    input wire logic b_valid_i,
    output logic b_ready_o,
    input wire logic [255:0] b_data_i,

    // results
    output logic y_valid_o,
    output logic [15:0] y_idx_o,
    output logic [31:0] y_o
);
  localparam int AW = $clog2(MaxA);
  localparam int LoopLat = 8;  // issue to issue of the same sums

  // ---------------------------------------------------------------------------
  // Job and beat position
  // ---------------------------------------------------------------------------
  logic run_q, mode_q;
  logic [8:0] hs_q;
  logic [15:0] nt_q;
  logic [15:0] outs_left_q;
  logic [31:0] beats_left_q;

  logic [4:0] c_q;  // beat within the row (mode 0: 0..15, mode 1: 0..hs/8-1)
  logic [15:0] u_q;  // mode 0: i; mode 1: t
  logic [8:0] blk_q;  // mode 0: block of 128 timesteps

  logic [4:0] c_last;
  logic [15:0] u_last;
  assign c_last = mode_q ? 5'(hs_q / 8 - 1) : 5'd15;
  assign u_last = mode_q ? nt_q - 1 : 16'(hs_q - 1);

  // the sums this beat adds to were not used in the last LoopLat-1 beats
  logic [LoopLat-2:0] recent_v;
  logic [LoopLat-2:0][4:0] recent_c;
  logic hazard;
  always_comb begin
    hazard = 1'b0;
    for (int k = 0; k < LoopLat - 1; k++)
      if (recent_v[k] && recent_c[k] == c_q) hazard = 1'b1;
  end

  logic issue;
  assign issue = run_q && b_valid_i && (beats_left_q != 0) && !hazard;
  assign b_ready_o = issue;
  assign busy_o = run_q;

  always_ff @(posedge clk_i, negedge rst_ni) begin
    if (~rst_ni) begin
      recent_v <= '0;
      recent_c <= '0;
    end else begin
      recent_v <= {recent_v[LoopLat-3:0], issue};
      recent_c <= {recent_c[LoopLat-3:0], c_q};
    end
  end

  logic out_fire;  // a result comes out this cycle

  always_ff @(posedge clk_i, negedge rst_ni) begin
    if (~rst_ni) begin
      run_q <= 1'b0;
      mode_q <= 1'b0;
      hs_q <= '0;
      nt_q <= '0;
      c_q <= '0;
      u_q <= '0;
      blk_q <= '0;
      beats_left_q <= '0;
      outs_left_q <= '0;
      done_o <= 1'b0;
    end else begin
      done_o <= 1'b0;
      if (start_i && !run_q) begin
        run_q <= 1'b1;
        mode_q <= mode_i;
        hs_q <= hs_i;
        nt_q <= nt_i;
        c_q <= '0;
        u_q <= '0;
        blk_q <= '0;
        beats_left_q <= beats_i;
        outs_left_q <= mode_i ? 16'(hs_i) : (nt_i + 16'd127) & ~16'd127;
      end else if (run_q) begin
        if (issue) begin
          beats_left_q <= beats_left_q - 1;
          if (c_q == c_last) begin
            c_q <= '0;
            if (u_q == u_last) begin
              u_q <= '0;
              blk_q <= blk_q + 1;
            end else begin
              u_q <= u_q + 1;
            end
          end else begin
            c_q <= c_q + 1;
          end
        end
        if (out_fire) begin
          outs_left_q <= outs_left_q - 1;
          if (outs_left_q == 1) begin
            run_q <= 1'b0;
            done_o <= 1'b1;
          end
        end
      end
    end
  end

  // ---------------------------------------------------------------------------
  // A RAM (block RAM, synchronous read): q[i] or a[t] for the whole beat
  // ---------------------------------------------------------------------------
  logic [31:0] a_ram[MaxA];
  logic [31:0] a_rdata;
  always_ff @(posedge clk_i) begin
    if (a_we_i) a_ram[a_addr_i] <= a_data_i;
    a_rdata <= a_ram[AW'(u_q)];
  end

  typedef struct packed {
    logic [4:0] acc;
    logic first;
    logic last;
    logic [15:0] out;  // index of lane 0's result
  } meta_t;

  meta_t e_meta;
  assign e_meta = '{
      acc: c_q,
      first: (u_q == 0),
      last: (u_q == u_last),
      out: mode_q ? 16'({c_q, 3'd0}) : 16'({blk_q, c_q[3:0], 3'd0})
  };

  // S1: A RAM data arrives; S2: registered into the multipliers
  logic s1_valid, s2_valid;
  meta_t s1_meta, s2_meta;
  logic [255:0] s1_b, s2_b;
  logic [31:0] s2_a;

  always_ff @(posedge clk_i, negedge rst_ni) begin
    if (~rst_ni) begin
      s1_valid <= 1'b0;
      s2_valid <= 1'b0;
    end else begin
      s1_valid <= issue;
      s2_valid <= s1_valid;
    end
  end

  always_ff @(posedge clk_i) begin
    s1_meta <= e_meta;
    s1_b    <= b_data_i;
    s2_meta <= s1_meta;
    s2_a    <= a_rdata;
    s2_b    <= s1_b;
  end

  // ---------------------------------------------------------------------------
  // Lanes: multiply, add into the lane's sum
  // ---------------------------------------------------------------------------
  logic [Lanes-1:0] lane_v;
  meta_t [Lanes-1:0] lane_meta;
  logic [Lanes-1:0][31:0] lane_y;

  for (genvar j = 0; j < Lanes; j++) begin : g_lane
    logic m_valid;
    meta_t m_meta;
    logic [31:0] m_y;

    fp32_mul #(
        .TagW($bits(meta_t))
    ) u_mul (
        .clk_i  (clk_i),
        .rst_ni (rst_ni),
        .valid_i(s2_valid),
        .tag_i  (s2_meta),
        .a_i    (s2_a),
        .b_i    (s2_b[32*j+:32]),
        .valid_o(m_valid),
        .tag_o  (m_meta),
        .y_o    (m_y)
    );

    // the lane's sums (distributed RAM: read here, written back when the
    // add is done)
    logic [31:0] acc_ram[32];

    fp32_add #(
        .TagW($bits(meta_t))
    ) u_add (
        .clk_i  (clk_i),
        .rst_ni (rst_ni),
        .valid_i(m_valid),
        .tag_i  (m_meta),
        .a_i    (m_meta.first ? 32'd0 : acc_ram[m_meta.acc]),
        .b_i    (m_y),
        .valid_o(lane_v[j]),
        .tag_o  (lane_meta[j]),
        .y_o    (lane_y[j])
    );

    always_ff @(posedge clk_i) begin
      if (lane_v[j]) acc_ram[lane_meta[j].acc] <= lane_y[j];
    end
  end

  // ---------------------------------------------------------------------------
  // Results: a beat's 8 sums at once into the FIFO, out one per cycle
  // ---------------------------------------------------------------------------
  logic fifo_empty, fifo_full, fifo_pop;
  logic [16+32*Lanes-1:0] fifo_rdata;

  fifo #(
      .Width(16 + 32 * Lanes),
      .Depth(32)
  ) u_out (
      .clk_i  (clk_i),
      .rst_ni (rst_ni),
      .clear_i(1'b0),
      .we_i   (lane_v[0] && lane_meta[0].last),
      .wdata_i({lane_meta[0].out, lane_y}),
      .re_i   (fifo_pop),
      .rdata_o(fifo_rdata),
      .empty_o(fifo_empty),
      .full_o (fifo_full)
  );

  logic [$clog2(Lanes)-1:0] k_q;  // lane being sent
  assign out_fire = !fifo_empty;
  assign fifo_pop = out_fire && (k_q == $clog2(Lanes)'(Lanes - 1));

  always_ff @(posedge clk_i, negedge rst_ni) begin
    if (~rst_ni) begin
      k_q <= '0;
      y_valid_o <= 1'b0;
    end else begin
      y_valid_o <= out_fire;
      if (out_fire) k_q <= k_q + 1;
    end
  end
  always_ff @(posedge clk_i) begin
    y_idx_o <= fifo_rdata[16+32*Lanes-1-:16] + 16'(k_q);
    y_o <= fifo_rdata[32*k_q+:32];
  end

endmodule

`default_nettype wire
