`default_nettype none

// Attention for one head, the two float loops of runq.c's forward(), with
// the same float operations in the same order (so the results are the same
// bits as on a CPU without FMA):
//
//   mode 0, scores:   y[t] = sum over i of q[i] * k_t[i]      (t < nt)
//   mode 1, weighted: y[i] = sum over t of a[t] * v_t[i]      (i < hs)
//
// each sum starting from +0 and adding its terms in index order. The CPU does
// the rest (the 1/sqrt(head_size) scaling and the softmax between the two).
//
// The A RAM holds q (mode 0) or the attention weights a (mode 1), written by
// the CPU. K or V streams in as 32-byte beats (8 floats), from DDR:
//   mode 0: K in blocks of Tb = 16 timesteps, each block [i][t % 16], so a
//           beat is 8 timesteps' element i; ceil(nt / 16) * hs * 2 beats
//   mode 1: V as rows [t][i]; nt * hs / 8 beats
// One multiply-add per cycle. Each sum is a chain of dependent adds; the
// order above keeps the 16 (mode 0) or hs (mode 1) sums in flight
// round-robin, more than the multiply-add loop's latency, so the chain never
// waits. hs must be a multiple of 8 and at least 16.
//
// Results come out on y_* (index, value) as each sum completes: mode 0 gives
// ceil(nt / 16) * 16 scores (the padding timesteps' are garbage), mode 1 hs.
module attn #(
    parameter int MaxA = 4096,   // A RAM words: max head size / timesteps
    parameter int MaxAcc = 256   // max head size
) (
    input wire logic clk_i,
    input wire logic rst_ni,

    input wire logic start_i,
    input wire logic mode_i,  // 0 scores, 1 weighted sum
    input wire logic [8:0] hs_i,
    input wire logic [15:0] nt_i,
    input wire logic [31:0] beats_i,  // mode 0: ceil(nt / 16) * hs * 2, mode 1: nt * hs / 8
    output logic busy_o,
    output logic done_o,  // one cycle, after the last result

    // A RAM (q or a), CPU writes
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
  localparam int CW = $clog2(MaxAcc);

  // ---------------------------------------------------------------------------
  // Job
  // ---------------------------------------------------------------------------
  logic run_q, mode_q;
  logic [8:0] hs_q;
  logic [15:0] nt_q;
  logic [15:0] outs_left_q;  // results still to come

  // element position (the element being issued this cycle)
  logic [2:0] j_q;      // lane within the beat
  logic [5:0] c_q;      // mode 0: half (0 / 1); mode 1: beat within the row
  logic [15:0] u_q;     // mode 0: i; mode 1: t
  logic [11:0] blk_q;   // mode 0: block of 16 timesteps
  logic [31:0] beats_left_q;

  logic [5:0] c_last;
  logic [15:0] u_last;
  assign c_last = mode_q ? 6'(hs_q / 8 - 1) : 6'd1;
  assign u_last = mode_q ? nt_q - 1 : 16'(hs_q - 1);

  logic issue;
  assign issue = run_q && b_valid_i && (beats_left_q != 0);
  assign b_ready_o = issue && (j_q == 3'd7);
  assign busy_o = run_q;

  // the element's fields
  logic [AW-1:0] e_a_idx;
  logic [CW-1:0] e_acc;
  logic e_first, e_last;
  logic [15:0] e_out;
  always_comb begin
    if (!mode_q) begin  // scores: a = q[i], sum r = half * 8 + j
      e_a_idx = AW'(u_q);
      e_acc   = CW'({c_q[0], j_q});
      e_first = (u_q == 0);
      e_last  = (u_q == u_last);
      e_out   = {blk_q, c_q[0], j_q};
    end else begin  // weighted sum: a = a[t], sum i = c * 8 + j
      e_a_idx = AW'(u_q);
      e_acc   = CW'({c_q, j_q});
      e_first = (u_q == 0);
      e_last  = (u_q == u_last);
      e_out   = 16'({c_q, j_q});
    end
  end

  logic out_fire;  // a result comes out this cycle

  always_ff @(posedge clk_i, negedge rst_ni) begin
    if (~rst_ni) begin
      run_q <= 1'b0;
      mode_q <= 1'b0;
      hs_q <= '0;
      nt_q <= '0;
      j_q <= '0;
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
        j_q <= '0;
        c_q <= '0;
        u_q <= '0;
        blk_q <= '0;
        beats_left_q <= beats_i;
        outs_left_q  <= mode_i ? 16'(hs_i) : (nt_i + 16'd15) & ~16'd15;
      end else if (run_q) begin
        if (issue) begin
          j_q <= j_q + 1;
          if (j_q == 3'd7) begin
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
  // A RAM (block RAM, synchronous read)
  // ---------------------------------------------------------------------------
  logic [31:0] a_ram[MaxA];
  logic [31:0] a_rdata;
  always_ff @(posedge clk_i) begin
    if (a_we_i) a_ram[a_addr_i] <= a_data_i;
    a_rdata <= a_ram[e_a_idx];
  end

  // S1: A RAM data arrives; S2: registered into the multiplier
  typedef struct packed {
    logic [CW-1:0] acc;
    logic first;
    logic last;
    logic [15:0] out;
  } meta_t;

  logic s1_valid, s2_valid;
  meta_t s1_meta, s2_meta;
  logic [31:0] s1_b, s2_a, s2_b;

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
    s1_meta <= '{acc: e_acc, first: e_first, last: e_last, out: e_out};
    s1_b    <= b_data_i[j_q*32+:32];
    s2_meta <= s1_meta;
    s2_a    <= a_rdata;
    s2_b    <= s1_b;
  end

  // ---------------------------------------------------------------------------
  // Multiply, then add into the element's sum
  // ---------------------------------------------------------------------------
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
      .b_i    (s2_b),
      .valid_o(m_valid),
      .tag_o  (m_meta),
      .y_o    (m_y)
  );

  // the sums (distributed RAM: read here, written back when the add is done)
  logic [31:0] acc_ram[MaxAcc];
  logic a_valid;
  meta_t a_meta;
  logic [31:0] a_y;

  fp32_add #(
      .TagW($bits(meta_t))
  ) u_add (
      .clk_i  (clk_i),
      .rst_ni (rst_ni),
      .valid_i(m_valid),
      .tag_i  (m_meta),
      .a_i    (m_meta.first ? 32'd0 : acc_ram[m_meta.acc]),
      .b_i    (m_y),
      .valid_o(a_valid),
      .tag_o  (a_meta),
      .y_o    (a_y)
  );

  always_ff @(posedge clk_i) begin
    if (a_valid) acc_ram[a_meta.acc] <= a_y;
  end

  assign out_fire = a_valid && a_meta.last;

  always_ff @(posedge clk_i, negedge rst_ni) begin
    if (~rst_ni) y_valid_o <= 1'b0;
    else y_valid_o <= out_fire;
  end
  always_ff @(posedge clk_i) begin
    y_idx_o <= a_meta.out;
    y_o <= a_y;
  end

endmodule

`default_nettype wire
