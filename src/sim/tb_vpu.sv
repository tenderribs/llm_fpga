`timescale 1ns / 1ps
`default_nettype none

// vpu.sv against sim/gen_vpu.py vectors (sim/run_vpu.sh): behavioral Y / V
// RAMs (registered address, then the read, as in llm_core), qmv x RAM and
// attn A RAM write capture; every result checked bit for bit.
module tb_vpu #(
    parameter string VecDir = "",
    parameter int Op = 1,  // 1 QUANT, 2 SILU, 3 SOFTMAX
    parameter int N = 32,
    parameter int NIn = 32,
    parameter int NOut = 2,
    parameter logic [31:0] FParam = 32'h0
);
  logic clk = 0, rst_n = 0;
  always #2 clk = ~clk;

  logic start, busy, done;
  logic [14:0] y_addr;
  logic [11:0] v_addr, a_addr;
  logic [5:0] xq_addr, xs_addr;
  logic y_we, v_we, xq_we, xs_we, a_we;
  logic [31:0] y_wdata, v_wdata, y_rdata, v_rdata, xs_data, a_data;
  logic [255:0] xq_data;

  // QUANT reads x from V (at 100), SILU a / b from Y (at 0 / 1000) into V
  // at 200, SOFTMAX scores from Y (at 0) with V at 300 as scratch
  localparam int SrcV = (Op == 1);
  localparam int Src = (Op == 1) ? 100 : 0;
  localparam int Src2 = 1000;
  localparam int Dst = (Op == 2) ? 200 : 300;

  vpu dut (
      .clk_i(clk), .rst_ni(rst_n),
      .start_i(start), .op_i(2'(Op)), .n_i(16'(N)), .src_v_i(1'(SrcV)), .src_i(16'(Src)),
      .src2_i(16'(Src2)), .dst_i(16'(Dst)), .fparam_i(FParam), .busy_o(busy), .done_o(done),
      .y_addr_o(y_addr), .y_we_o(y_we), .y_wdata_o(y_wdata), .y_rdata_i(y_rdata),
      .v_addr_o(v_addr), .v_we_o(v_we), .v_wdata_o(v_wdata), .v_rdata_i(v_rdata),
      .xq_we_o(xq_we), .xq_addr_o(xq_addr), .xq_data_o(xq_data),
      .xs_we_o(xs_we), .xs_addr_o(xs_addr), .xs_data_o(xs_data),
      .a_we_o(a_we), .a_addr_o(a_addr), .a_data_o(a_data)
  );

  logic [31:0] yram[32768];
  logic [31:0] vram[4096];
  logic [255:0] xq[64];
  logic [31:0] xs[64];
  logic [31:0] aram[4096];
  // like llm_core: address, write enable and data registered, then the RAM
  logic y_we_q, v_we_q;
  logic [14:0] y_addr_q;
  logic [11:0] v_addr_q;
  logic [31:0] y_wdata_q, v_wdata_q;
  always_ff @(posedge clk) begin
    y_we_q <= y_we;
    y_addr_q <= y_addr;
    y_wdata_q <= y_wdata;
    v_we_q <= v_we;
    v_addr_q <= v_addr;
    v_wdata_q <= v_wdata;
    if (y_we_q) yram[y_addr_q] <= y_wdata_q;
    y_rdata <= yram[y_addr_q];
    if (v_we_q) vram[v_addr_q] <= v_wdata_q;
    v_rdata <= vram[v_addr_q];
    if (xq_we) xq[xq_addr] <= xq_data;
    if (xs_we) xs[xs_addr] <= xs_data;
    if (a_we) aram[a_addr] <= a_data;
  end

  logic [31:0] in_w[NIn];
  logic [255:0] out_l[64];
  logic [31:0] out_w[NOut];
  int errors = 0;

  initial begin
    $readmemh({VecDir, "/in.hex"}, in_w);
    if (Op == 1) $readmemh({VecDir, "/out.hex"}, out_l, 0, N / 32 - 1);
    $readmemh({VecDir, "/out.hex"}, out_w);
    for (int i = 0; i < NIn; i++) begin
      if (Op == 1) vram[Src+i] = in_w[i];
      else if (Op == 2) begin
        if (i < N) yram[Src+i] = in_w[i];
        else yram[Src2+i-N] = in_w[i];
      end else yram[Src+i] = in_w[i];
    end
    start = 0;
    repeat (5) @(posedge clk);
    rst_n = 1;
    @(negedge clk);
    start = 1;
    @(negedge clk);
    start = 0;
    for (int c = 0; !done; c++) begin
      if (c == 200000) begin
        $display("TIMEOUT");
        $finish;
      end
      @(posedge clk);
    end
    repeat (3) @(posedge clk);
    if (busy) begin
      $display("MISMATCH still busy after done");
      errors++;
    end
    if (Op == 1) begin
      for (int g = 0; g < N / 32; g++) begin
        // out.hex: N/32 lines (64 hex digits each), then N/32 scales
        if (xq[g] !== out_l[g]) begin
          if (errors < 6) $display("MISMATCH xq line %0d: got %h, exp %h", g, xq[g], out_l[g]);
          errors++;
        end
        if (xs[g] !== out_w[N/32+g]) begin
          if (errors < 6) $display("MISMATCH xs[%0d]: got %08h, exp %08h", g, xs[g], out_w[N/32+g]);
          errors++;
        end
      end
    end else begin
      for (int i = 0; i < N; i++) begin
        logic [31:0] got;
        got = (Op == 2) ? vram[Dst+i] : aram[i];
        if (got !== out_w[i]) begin
          if (errors < 6) $display("MISMATCH [%0d]: got %08h, exp %08h", i, got, out_w[i]);
          errors++;
        end
      end
    end
    $display("TEST vpu op %0d n %0d", Op, N);
    if (errors == 0) $display("TEST PASSED");
    else $display("FAIL: %0d mismatches", errors);
    $finish;
  end
endmodule

`default_nettype wire
