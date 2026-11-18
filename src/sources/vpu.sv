`default_nettype none

// Vector unit: the float work between the matmuls and the attention unit,
// with the firmware's float operations in the firmware's order (fw/model,
// runq.c), so the results are the same bits:
//
//   COPY     A RAM[i] = V[src + i]                    (q for the attention unit)
//   QUANT    x[0..n) (Y or V RAM) -> int8 groups of 32 into qmv's x RAM:
//            per group wmax = max |x|, scale = wmax / 127,
//            q = round-half-away(x / scale); xs = scale
//   SILU     V[dst + i] = (a * (1 / (1 + e^-a))) * b,  a = Y[src + i],
//            b = Y[src2 + i]                          (SwiGLU)
//   SOFTMAX  s_i = Y[src + i] / fparam (into V[dst + i]); max; e_i =
//            e^(s_i - max) (into Y[i]); sum = 0 + e_0 + e_1 + ... in order;
//            A RAM[i] = e_i / sum                     (attention weights)
//
// Each operation runs as passes that stream elements through shared units
// (one divider, one exp, one adder, two multipliers): read data is used
// RdLat cycles after the address goes out (the RAMs register the address,
// read, and the data is registered here), results are written as they come
// out. The
// softmax sum is a chain of dependent adds, one element per add latency.
// Exp results below 2^-126 are flushed to zero (see fp32_exp).
module vpu #(
    parameter int YAddrW = 15,
    parameter int VAddrW = 13,
    parameter int XLineW = 6,  // qmv x RAM lines (MaxN / 32)
    parameter int AAddrW = 12,
    parameter int RdLat = 3
) (
    input wire logic clk_i,
    input wire logic rst_ni,

    input wire logic start_i,
    input wire logic [1:0] op_i,  // 0 COPY, 1 QUANT, 2 SILU, 3 SOFTMAX
    input wire logic [15:0] n_i,
    input wire logic src_v_i,  // QUANT: source in V RAM (else Y RAM)
    input wire logic [15:0] src_i,
    input wire logic [15:0] src2_i,
    input wire logic [15:0] dst_i,
    input wire logic [31:0] fparam_i,
    output logic busy_o,
    output logic done_o,

    // Y RAM: one port, read (data next cycle) or write
    output logic [YAddrW-1:0] y_addr_o,
    output logic y_we_o,
    output logic [31:0] y_wdata_o,
    input wire logic [31:0] y_rdata_i,
    // V RAM: one port, read (data next cycle) or write
    output logic [VAddrW-1:0] v_addr_o,
    output logic v_we_o,
    output logic [31:0] v_wdata_o,
    input wire logic [31:0] v_rdata_i,
    // qmv's x RAM
    output logic xq_we_o,
    output logic [XLineW-1:0] xq_addr_o,
    output logic [255:0] xq_data_o,
    output logic xs_we_o,
    output logic [XLineW-1:0] xs_addr_o,
    output logic [31:0] xs_data_o,
    // attn's A RAM
    output logic a_we_o,
    output logic [AAddrW-1:0] a_addr_o,
    output logic [31:0] a_data_o
);
  localparam logic [1:0] OP_COPY = 2'd0, OP_QUANT = 2'd1, OP_SILU = 2'd2, OP_SOFTMAX = 2'd3;
  localparam logic [31:0] F_ONE = 32'h3F80_0000, F_127 = 32'h42FE_0000;

  typedef enum logic [3:0] {
    P_IDLE, P_COPY, P_QMAX, P_QSCALE, P_QDIV, P_SILU, P_SM1, P_SM2, P_SM3, P_SM4
  } pass_t;
  pass_t pass_q;

  logic [15:0] n_q, src_q, src2_q, dst_q;
  logic src_v_q;
  logic [31:0] fparam_q;

  // ordered key of a float (-0 as +0): a > b as floats iff key(a) > key(b)
  function automatic logic [31:0] fkey(input logic [31:0] f);
    if (f[30:0] == '0) return 32'h8000_0000;
    return f[31] ? ~f : {1'b1, f[30:0]};
  endfunction

  // ---------------------------------------------------------------------------
  // Element issue: a read per element (SILU: two), data back next cycle
  // ---------------------------------------------------------------------------
  logic [15:0] iss_q;  // next element to issue
  logic [15:0] res_q;  // results of this pass so far
  logic ph_q;  // SILU: 0 read a, 1 read b
  // reads in flight: RdLat stages; the last is the registered data's cycle
  logic [RdLat-1:0] rd_v_sr, rd_ph_sr;
  logic [RdLat-1:0][15:0] rd_i_sr;
  logic rd_v_q, rd_ph_q;  // the data in d_q is element rd_i_q's
  logic [15:0] rd_i_q;
  logic [31:0] d_q;
  assign rd_v_q = rd_v_sr[RdLat-1];
  assign rd_ph_q = rd_ph_sr[RdLat-1];
  assign rd_i_q = rd_i_sr[RdLat-1];
  logic sum_busy_q;  // SM3: an add in flight

  logic [15:0] limit;  // elements this pass
  always_comb begin
    case (pass_q)
      P_QMAX, P_QDIV: limit = 16'd32;
      default: limit = n_q;
    endcase
  end

  logic issue;
  always_comb begin
    case (pass_q)
      P_COPY, P_QMAX, P_QDIV, P_SILU, P_SM1, P_SM2, P_SM4: issue = (iss_q < limit);
      P_SM3: issue = (iss_q < limit) && !sum_busy_q && (rd_v_sr == '0);
      default: issue = 1'b0;
    endcase
  end

  logic [5:0] grp_q;  // QUANT: group
  logic [15:0] rd_addr;  // element's source address
  always_comb begin
    case (pass_q)
      P_QMAX, P_QDIV: rd_addr = src_q + 16'({grp_q, 5'd0}) + iss_q;
      P_SILU: rd_addr = (ph_q ? src2_q : src_q) + iss_q;
      P_SM1, P_COPY: rd_addr = src_q + iss_q;
      P_SM2: rd_addr = dst_q + iss_q;  // V
      default: rd_addr = iss_q;  // SM3, SM4: Y[i]
    endcase
  end

  logic rd_from_v;  // this pass reads the V RAM
  assign rd_from_v = (pass_q == P_SM2) || (pass_q == P_COPY) ||
                     ((pass_q == P_QMAX || pass_q == P_QDIV) && src_v_q);

  logic [31:0] rd_data;
  assign rd_data = d_q;
  always_ff @(posedge clk_i) d_q <= rd_from_v ? v_rdata_i : y_rdata_i;

  // ---------------------------------------------------------------------------
  // Units. Tag: element index, a, b (SILU's operands)
  // ---------------------------------------------------------------------------
  typedef struct packed {
    logic [15:0] i;
    logic [31:0] a;
    logic [31:0] b;
  } tag_t;

  logic div_v, exp_v, add_v, mul1_v, mul2_v;
  tag_t div_t, exp_t, add_t, mul1_t, mul2_t;
  logic [31:0] div_y, exp_y, add_y, mul1_y, mul2_y;

  logic div_iv, exp_iv, add_iv;
  tag_t div_it, exp_it, add_it;
  logic [31:0] div_a, div_b, exp_x, add_a, add_b;

  logic [31:0] wmax_q, scale_q, max_q, sum_q, a_hold_q;

  always_comb begin
    div_iv = 1'b0;
    div_it = '0;
    div_a = '0;
    div_b = '0;
    exp_iv = 1'b0;
    exp_it = '0;
    exp_x = '0;
    add_iv = 1'b0;
    add_it = '0;
    add_a = '0;
    add_b = '0;
    case (pass_q)
      P_QDIV: begin
        div_iv = rd_v_q;
        div_it.i = rd_i_q;
        div_a = rd_data;
        div_b = scale_q;
      end
      P_SILU: begin
        // a and b in: e^-a; out of exp: 1 + e; out of the add: 1 / (1 + e)
        exp_iv = rd_v_q && rd_ph_q;
        exp_it = '{i: rd_i_q, a: a_hold_q, b: rd_data};
        exp_x = {~a_hold_q[31], a_hold_q[30:0]};
        add_iv = exp_v;
        add_it = exp_t;
        add_a = F_ONE;
        add_b = exp_y;
        div_iv = add_v;
        div_it = add_t;
        div_a = F_ONE;
        div_b = add_y;
      end
      P_SM1: begin
        div_iv = rd_v_q;
        div_it.i = rd_i_q;
        div_a = rd_data;
        div_b = fparam_q;
      end
      P_SM2: begin
        add_iv = rd_v_q;
        add_it.i = rd_i_q;
        add_a = rd_data;
        add_b = {~max_q[31], max_q[30:0]};
        exp_iv = add_v;
        exp_it = add_t;
        exp_x = add_y;
      end
      P_SM3: begin
        add_iv = rd_v_q;
        add_a = sum_q;
        add_b = rd_data;
      end
      P_SM4: begin
        div_iv = rd_v_q;
        div_it.i = rd_i_q;
        div_a = rd_data;
        div_b = sum_q;
      end
      default: ;
    endcase
  end

  // the one-off division of QSCALE goes in on its own
  logic qscale_go;
  logic div_iv_all;
  logic [31:0] div_a_all, div_b_all;
  assign div_iv_all = div_iv || qscale_go;
  assign div_a_all = qscale_go ? wmax_q : div_a;
  assign div_b_all = qscale_go ? F_127 : div_b;

  fp32_div #(.TagW($bits(tag_t))) u_div (
      .clk_i, .rst_ni, .valid_i(div_iv_all), .tag_i(div_it), .a_i(div_a_all), .b_i(div_b_all),
      .valid_o(div_v), .tag_o(div_t), .y_o(div_y)
  );
  fp32_exp #(.TagW($bits(tag_t))) u_exp (
      .clk_i, .rst_ni, .valid_i(exp_iv), .tag_i(exp_it), .x_i(exp_x),
      .valid_o(exp_v), .tag_o(exp_t), .y_o(exp_y)
  );
  fp32_add #(.TagW($bits(tag_t))) u_add (
      .clk_i, .rst_ni, .valid_i(add_iv), .tag_i(add_it), .a_i(add_a), .b_i(add_b),
      .valid_o(add_v), .tag_o(add_t), .y_o(add_y)
  );
  // SILU: a * (1 / (1 + e)), then * b
  fp32_mul #(.TagW($bits(tag_t))) u_mul1 (
      .clk_i, .rst_ni, .valid_i(div_v && pass_q == P_SILU), .tag_i(div_t),
      .a_i(div_t.a), .b_i(div_y), .valid_o(mul1_v), .tag_o(mul1_t), .y_o(mul1_y)
  );
  fp32_mul #(.TagW($bits(tag_t))) u_mul2 (
      .clk_i, .rst_ni, .valid_i(mul1_v), .tag_i(mul1_t),
      .a_i(mul1_y), .b_i(mul1_t.b), .valid_o(mul2_v), .tag_o(mul2_t), .y_o(mul2_y)
  );

  // round half away from zero to int8 (|v| <= 127.5 here; NaN, from 0 / 0,
  // gives -1 like RISC-V's fcvt.w.s + truncation)
  function automatic logic [7:0] rmm8(input logic [31:0] v);
    logic [7:0] e;
    logic [31:0] fx, mag;
    e = v[30:23];
    if (e == 8'hFF) return 8'hFF;
    if (e < 8'd126) return 8'd0;
    fx = 32'({1'b1, v[22:0]}) << (e - 8'd126);  // |v| * 2^24
    mag = (fx + 32'h0080_0000) >> 24;  // + 0.5, truncate
    return v[31] ? 8'(-mag) : 8'(mag);
  endfunction

  // ---------------------------------------------------------------------------
  // Writes
  // ---------------------------------------------------------------------------
  logic [31:0][7:0] line_q;
  logic v_wr, y_wr;
  logic [15:0] v_waddr, y_waddr;
  logic [31:0] v_wdata, y_wdata;

  always_comb begin
    v_wr = 1'b0;
    v_waddr = '0;
    v_wdata = '0;
    y_wr = 1'b0;
    y_waddr = '0;
    y_wdata = '0;
    case (pass_q)
      P_SILU: begin
        v_wr = mul2_v;
        v_waddr = dst_q + mul2_t.i;
        v_wdata = mul2_y;
      end
      P_SM1: begin
        v_wr = div_v;
        v_waddr = dst_q + div_t.i;
        v_wdata = div_y;
      end
      P_SM2: begin
        y_wr = exp_v;
        y_waddr = exp_t.i;
        y_wdata = exp_y;
      end
      default: ;
    endcase
  end

  assign v_addr_o  = VAddrW'(v_wr ? v_waddr : rd_addr);
  assign v_we_o    = v_wr;
  assign v_wdata_o = v_wdata;
  assign y_addr_o  = YAddrW'(y_wr ? y_waddr : rd_addr);
  assign y_we_o    = y_wr;
  assign y_wdata_o = y_wdata;

  logic copy_wr;
  assign copy_wr  = (pass_q == P_COPY) && rd_v_q;
  assign a_we_o   = ((pass_q == P_SM4) && div_v) || copy_wr;
  assign a_addr_o = AAddrW'(copy_wr ? rd_i_q : div_t.i);
  assign a_data_o = copy_wr ? rd_data : div_y;

  assign busy_o = (pass_q != P_IDLE);

  // ---------------------------------------------------------------------------
  // Pass sequencing
  // ---------------------------------------------------------------------------
  // QDIV: the divider's output registered, then rounded into the line
  logic dq_v;
  logic [4:0] dq_i;
  logic [31:0] dq_y;
  always_ff @(posedge clk_i, negedge rst_ni) begin
    if (~rst_ni) dq_v <= 1'b0;
    else dq_v <= (pass_q == P_QDIV) && div_v;
  end
  always_ff @(posedge clk_i) begin
    dq_i <= div_t.i[4:0];
    dq_y <= div_y;
  end
  logic last_line;  // the line's last byte is in dq now
  assign last_line = (pass_q == P_QDIV) && dq_v && (res_q == 16'd31);

  // SM1: the result and its ordered key registered, then the max update
  logic mk_v;
  logic [15:0] mk_i;
  logic [31:0] mk_y, mk_key, max_key_q;
  always_ff @(posedge clk_i, negedge rst_ni) begin
    if (~rst_ni) mk_v <= 1'b0;
    else mk_v <= (pass_q == P_SM1) && div_v;
  end
  always_ff @(posedge clk_i) begin
    mk_i <= div_t.i;
    mk_y <= div_y;
    mk_key <= fkey(div_y);
  end

  always_ff @(posedge clk_i, negedge rst_ni) begin
    if (~rst_ni) begin
      pass_q <= P_IDLE;
      done_o <= 1'b0;
      iss_q <= '0;
      res_q <= '0;
      ph_q <= 1'b0;
      rd_v_sr <= '0;
      rd_ph_sr <= '0;
      rd_i_sr <= '0;
      grp_q <= '0;
      sum_busy_q <= 1'b0;
      qscale_go <= 1'b0;
      xq_we_o <= 1'b0;
      xs_we_o <= 1'b0;
    end else begin
      done_o <= 1'b0;
      qscale_go <= 1'b0;
      xq_we_o <= 1'b0;
      xs_we_o <= 1'b0;

      // issue
      rd_v_sr <= {rd_v_sr[RdLat-2:0], issue};
      rd_ph_sr <= {rd_ph_sr[RdLat-2:0], ph_q};
      rd_i_sr <= {rd_i_sr[RdLat-2:0], iss_q};
      if (issue) begin
        if (pass_q == P_SILU) begin
          ph_q <= !ph_q;
          if (ph_q) iss_q <= iss_q + 1;
        end else begin
          iss_q <= iss_q + 1;
        end
      end

      case (pass_q)
        P_IDLE: begin
          if (start_i) begin
            n_q <= n_i;
            src_q <= src_i;
            src2_q <= src2_i;
            dst_q <= dst_i;
            src_v_q <= src_v_i;
            fparam_q <= fparam_i;
            iss_q <= '0;
            res_q <= '0;
            ph_q <= 1'b0;
            grp_q <= '0;
            wmax_q <= '0;
            sum_q <= '0;
            case (op_i)
              OP_COPY: pass_q <= P_COPY;
              OP_QUANT: pass_q <= P_QMAX;
              OP_SILU: pass_q <= P_SILU;
              default: pass_q <= P_SM1;  // OP_SOFTMAX
            endcase
          end
        end

        P_COPY: begin
          if (rd_v_q) begin
            res_q <= res_q + 1;
            if (res_q == n_q - 1) begin
              pass_q <= P_IDLE;
              done_o <= 1'b1;
            end
          end
        end

        P_QMAX: begin
          if (rd_v_q) begin
            if (rd_data[30:0] > wmax_q[30:0]) wmax_q <= {1'b0, rd_data[30:0]};
            res_q <= res_q + 1;
            if (res_q == 16'd31) begin
              qscale_go <= 1'b1;
              pass_q <= P_QSCALE;
            end
          end
        end

        P_QSCALE: begin
          if (div_v) begin
            scale_q <= div_y;
            iss_q <= '0;
            res_q <= '0;
            pass_q <= P_QDIV;
          end
        end

        P_QDIV: begin
          if (dq_v) begin
            line_q[dq_i] <= rmm8(dq_y);
            res_q <= res_q + 1;
          end
          if (last_line) begin
            xq_we_o <= 1'b1;
            xq_addr_o <= XLineW'(grp_q);
            xq_data_o <= {rmm8(dq_y), line_q[30:0]};
            xs_we_o <= 1'b1;
            xs_addr_o <= XLineW'(grp_q);
            xs_data_o <= scale_q;
            grp_q <= grp_q + 1;
            iss_q <= '0;
            res_q <= '0;
            wmax_q <= '0;
            if (16'({grp_q, 5'd0}) + 16'd32 >= n_q) begin
              pass_q <= P_IDLE;
              done_o <= 1'b1;
            end else begin
              pass_q <= P_QMAX;
            end
          end
        end

        P_SILU: begin
          if (rd_v_q && !rd_ph_q) a_hold_q <= rd_data;
          if (mul2_v) begin
            res_q <= res_q + 1;
            if (res_q == n_q - 1) begin
              pass_q <= P_IDLE;
              done_o <= 1'b1;
            end
          end
        end

        P_SM1: begin
          if (mk_v) begin
            if (mk_i == 0 || mk_key > max_key_q) begin
              max_q <= mk_y;
              max_key_q <= mk_key;
            end
            res_q <= res_q + 1;
            if (res_q == n_q - 1) begin
              iss_q <= '0;
              res_q <= '0;
              pass_q <= P_SM2;
            end
          end
        end

        P_SM2: begin
          if (exp_v) begin
            res_q <= res_q + 1;
            if (res_q == n_q - 1) begin
              iss_q <= '0;
              res_q <= '0;
              sum_q <= '0;
              pass_q <= P_SM3;
            end
          end
        end

        P_SM3: begin
          if (rd_v_q) sum_busy_q <= 1'b1;
          if (add_v) begin
            sum_q <= add_y;
            sum_busy_q <= 1'b0;
            res_q <= res_q + 1;
            if (res_q == n_q - 1) begin
              iss_q <= '0;
              res_q <= '0;
              pass_q <= P_SM4;
            end
          end
        end

        P_SM4: begin
          if (div_v) begin
            res_q <= res_q + 1;
            if (res_q == n_q - 1) begin
              pass_q <= P_IDLE;
              done_o <= 1'b1;
            end
          end
        end

        default: pass_q <= P_IDLE;
      endcase
    end
  end

endmodule

`default_nettype wire
