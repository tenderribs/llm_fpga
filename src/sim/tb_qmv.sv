`timescale 1ns / 1ps

// Runs one real runq.c matmul through qmv (vectors from sim/gen_qmv.py mm) and
// checks every output bit-exact. The weight stream has random gaps, so the
// pipeline sees stalls at every point in the beat sequence.
module tb_qmv #(
    parameter string VecDir,
    parameter int N,  // matmul inputs
    parameter int D,  // matmul outputs
    parameter int GS,
    parameter int P = 32,
    parameter int R = 16
) ();

  localparam int Beats = D * N / P;
  localparam int MaxN = 2048;

  logic clk, rst_n;
  initial begin
    clk = 0;
    forever #2.5 clk = ~clk;
  end

  logic [31:0] cfg[2];
  logic [P*8-1:0] xq[N/P];
  logic [31:0] xs[N/GS];
  logic [P*8+31:0] w[Beats];
  logic [31:0] y_exp[D];

  logic start, busy, done;
  logic xq_we, xs_we;
  logic [$clog2(MaxN/P)-1:0] xq_addr;
  logic [$clog2(MaxN/GS)-1:0] xs_addr;
  logic [P*8-1:0] xq_data;
  logic [31:0] xs_data;
  logic w_valid, w_ready;
  logic [P*8-1:0] w_q;
  logic [31:0] w_s;
  logic y_valid;
  logic [31:0] y;

  qmv #(
      .P(P),
      .GS(GS),
      .R(R),
      .MaxN(MaxN)
  ) dut (
      .clk_i(clk),
      .rst_ni(rst_n),
      .abort_i(1'b0),
      .start_i(start),
      .n_i(16'(N)),
      .d_i(16'(D)),
      .busy_o(busy),
      .done_o(done),
      .xq_we_i(xq_we),
      .xq_addr_i(xq_addr),
      .xq_data_i(xq_data),
      .xs_we_i(xs_we),
      .xs_addr_i(xs_addr),
      .xs_data_i(xs_data),
      .w_valid_i(w_valid),
      .w_ready_o(w_ready),
      .w_q_i(w_q),
      .w_s_i(w_s),
      .y_valid_o(y_valid),
      .y_o(y)
  );

  initial begin
    repeat (Beats * 3 + 10_000) @(posedge clk);
    $display("TIMEOUT: %0d of %0d rows out, %0d beats sent, busy=%b w_ready=%b",
             rows, D, beats_sent, busy, w_ready);
    $fatal;
  end

  int rows = 0;
  int errors = 0;
  int dones = 0;
  int beats_sent = 0;

  always @(posedge clk) begin
    if (rst_n && y_valid) begin
      if (rows < D && y !== y_exp[rows]) begin
        if (errors < 10) $display("MISMATCH y[%0d]: got %08h, exp %08h", rows, y, y_exp[rows]);
        errors++;
      end
      rows++;
    end
    if (rst_n && done) dones++;
  end

  initial begin
    $display("TEST qmv N=%0d D=%0d GS=%0d", N, D, GS);
    $readmemh({VecDir, "/cfg.hex"}, cfg);
    $readmemh({VecDir, "/xq.hex"}, xq);
    $readmemh({VecDir, "/xs.hex"}, xs);
    $readmemh({VecDir, "/w.hex"}, w);
    $readmemh({VecDir, "/y.hex"}, y_exp);
    if (cfg[0] != N || cfg[1] != D) $fatal(1, "vector size %0dx%0d != params", cfg[1], cfg[0]);

    rst_n = 0;
    start = 0;
    xq_we = 0;
    xs_we = 0;
    w_valid = 0;
    repeat (5) @(posedge clk);
    rst_n <= 1;
    @(posedge clk);

    for (int i = 0; i < N / P; i++) begin
      xq_we <= 1;
      xq_addr <= ($clog2(MaxN/P))'(i);
      xq_data <= xq[i];
      @(posedge clk);
    end
    xq_we <= 0;
    for (int i = 0; i < N / GS; i++) begin
      xs_we <= 1;
      xs_addr <= ($clog2(MaxN/GS))'(i);
      xs_data <= xs[i];
      @(posedge clk);
    end
    xs_we <= 0;

    start <= 1;
    @(posedge clk);
    start <= 0;

    // Drive the stream on the falling edge, where w_ready is stable: a beat
    // set up at a falling edge with w_ready high is taken at the next rising
    // edge. (Checking w_ready at the rising edge itself races with the DUT
    // dropping it after the last beat.)
    for (int i = 0; i < Beats; i++) begin
      // stall on ~1 in 4 cycles; long stalls now and then
      while ($urandom_range(0, 3) == 0) begin
        @(negedge clk);
        w_valid = 0;
        repeat ($urandom_range(0, 15) == 0 ? 20 : 0) @(negedge clk);
      end
      @(negedge clk);
      while (!w_ready) begin
        w_valid = 0;
        @(negedge clk);
      end
      w_valid = 1;
      w_q = w[i][P*8-1:0];
      w_s = w[i][P*8+:32];
      @(posedge clk);
      beats_sent++;
    end
    @(negedge clk);
    w_valid = 0;

    wait (!busy);
    repeat (50) @(posedge clk);

    if (rows != D) begin
      $display("MISMATCH row count: got %0d, exp %0d", rows, D);
      errors++;
    end
    if (dones != 1) begin
      $display("MISMATCH done pulses: got %0d, exp 1", dones);
      errors++;
    end
    if (errors == 0) $display("TEST PASSED");
    else $display("FAIL: %0d mismatches", errors);
    $finish;
  end
endmodule
