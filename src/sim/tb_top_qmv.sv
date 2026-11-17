`timescale 1ns / 1ps

// End-to-end test of top_qmv through the UART pins, on a real runq.c matmul
// (vectors from sim/gen_qmv.py mm). The UART runs at 8 clocks per bit.
//
// Sequence:
//   1. half a header, then silence: the idle timeout must discard it
//   2. one full job; every result checked bit-exact
module tb_top_qmv #(
    parameter string VecDir,
    parameter int N,
    parameter int D,
    parameter int GS,
    parameter int P = 32
) ();

  localparam int ClkHz = 200_000_000;
  localparam int ClksPerBit = 8;
  localparam int IdleTimeoutClks = 40 * ClksPerBit;
  localparam int Beats = D * N / P;

  logic clk;
  logic cpu_resetn;
  logic uart_rx;
  logic uart_tx;
  logic [7:0] led;

  logic [P*8-1:0] xq[N/P];
  logic [31:0] xs[N/GS];
  logic [P*8+31:0] w[Beats];
  logic [31:0] y_exp[D];

  top_qmv #(
      .ClkHz(ClkHz),
      .Baud(ClkHz / ClksPerBit),
      .IdleTimeoutClks(IdleTimeoutClks),
      .P(P),
      .GS(GS)
  ) dut (
      .sysclk_p  (clk),
      .sysclk_n  (~clk),
      .cpu_resetn(cpu_resetn),
      .uart_rx_i (uart_rx),
      .uart_tx_o (uart_tx),
      .led_o     (led)
  );

  initial begin
    clk = 0;
    forever #2.5 clk = ~clk;
  end

  initial begin
    cpu_resetn = 0;
    repeat (10) @(posedge clk);
    cpu_resetn = 1;
  end

  initial begin
    repeat ((Beats * (P + 4) + N * 2 + D * 4 + 100) * 10 * ClksPerBit + 100_000) @(posedge clk);
    $display("TIMEOUT");
    $fatal;
  end

  task automatic send_byte(input logic [7:0] d);
    uart_rx = 1'b0;
    repeat (ClksPerBit) @(posedge clk);
    for (int i = 0; i < 8; i++) begin
      uart_rx = d[i];
      repeat (ClksPerBit) @(posedge clk);
    end
    uart_rx = 1'b1;
    repeat (ClksPerBit) @(posedge clk);
  endtask

  task automatic send_u32(input logic [31:0] v);
    for (int i = 0; i < 4; i++) send_byte(v[i*8+:8]);
  endtask

  logic [7:0] rx_bytes[$];
  int framing_errors = 0;

  initial begin
    logic [7:0] d;
    @(posedge clk iff cpu_resetn);
    forever begin
      @(negedge uart_tx);
      repeat (ClksPerBit / 2) @(posedge clk);
      for (int i = 0; i < 8; i++) begin
        repeat (ClksPerBit) @(posedge clk);
        d[i] = uart_tx;
      end
      repeat (ClksPerBit) @(posedge clk);
      if (uart_tx !== 1'b1) framing_errors++;
      rx_bytes.push_back(d);
    end
  end

  initial begin
    static int errors = 0;
    logic [31:0] y_got;

    $display("TEST top_qmv N=%0d D=%0d GS=%0d", N, D, GS);
    $readmemh({VecDir, "/xq.hex"}, xq);
    $readmemh({VecDir, "/xs.hex"}, xs);
    $readmemh({VecDir, "/w.hex"}, w);
    $readmemh({VecDir, "/y.hex"}, y_exp);

    uart_rx = 1'b1;
    @(posedge clk iff cpu_resetn);
    repeat (10) @(posedge clk);

    // 1. partial header, then silence
    send_byte(8'h12);
    send_byte(8'h34);
    repeat (2 * IdleTimeoutClks) @(posedge clk);

    // 2. full job
    send_byte(8'(N));
    send_byte(8'(N >> 8));
    send_byte(8'(D));
    send_byte(8'(D >> 8));
    for (int i = 0; i < N / P; i++) for (int b = 0; b < P; b++) send_byte(xq[i][b*8+:8]);
    for (int i = 0; i < N / GS; i++) send_u32(xs[i]);
    for (int i = 0; i < Beats; i++) begin
      for (int b = 0; b < P; b++) send_byte(w[i][b*8+:8]);
      send_u32(w[i][P*8+:32]);
    end

    wait (rx_bytes.size() == 4 * D);
    repeat (20 * ClksPerBit) @(posedge clk);

    for (int i = 0; i < D; i++) begin
      y_got = {rx_bytes[4*i+3], rx_bytes[4*i+2], rx_bytes[4*i+1], rx_bytes[4*i]};
      if (y_got !== y_exp[i]) begin
        if (errors < 10) $display("MISMATCH y[%0d]: got %08h, exp %08h", i, y_got, y_exp[i]);
        errors++;
      end
    end
    if (rx_bytes.size() != 4 * D) begin
      $display("MISMATCH byte count: got %0d, exp %0d", rx_bytes.size(), 4 * D);
      errors++;
    end
    if (framing_errors != 0) begin
      $display("MISMATCH %0d UART framing errors", framing_errors);
      errors++;
    end
    if (led[7:4] !== 4'd1) begin
      $display("MISMATCH job counter LEDs: got %0d, exp 1", led[7:4]);
      errors++;
    end
    if (errors == 0) $display("TEST PASSED");
    else $display("FAIL: %0d mismatches", errors);
    $finish;
  end
endmodule
