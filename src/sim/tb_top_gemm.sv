`timescale 1ns / 1ps

// End-to-end test of top_gemm: drives A, B, C into the UART RX pin, decodes Y from
// the UART TX pin and compares it to the golden vectors. The baud rate is set
// to 8 clocks per bit so the sim runs quickly.
//
// Sequence:
//   1. a few stray bytes, then silence: the idle timeout must discard them
//   2. two full jobs back to back, both checked
module tb_top_gemm #(
    parameter int N,
    parameter string AFile,
    parameter string BFile,
    parameter string CFile,
    parameter string YFile,
    parameter string FstPath
) ();

  localparam int ClkHz = 200_000_000;
  localparam int ClksPerBit = 8;
  localparam int IdleTimeoutClks = 40 * ClksPerBit;
  localparam int NumJobs = 2;
  localparam int OutBytes = 4 * N * N;

  logic clk;
  logic cpu_resetn;
  logic uart_rx;  // into the DUT
  logic uart_tx;  // out of the DUT
  logic [7:0] led;

  logic signed [7:0] a[N][N];
  logic signed [7:0] b[N][N];
  logic signed [7:0] c[N][N];
  logic signed [31:0] y_exp[N][N];

  top_gemm #(
      .N(N),
      .ClkHz(ClkHz),
      .Baud(ClkHz / ClksPerBit),
      .IdleTimeoutClks(IdleTimeoutClks)
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
    forever #2.5 clk = ~clk;  // 200 MHz
  end

  initial begin
    cpu_resetn = 0;
    repeat (10) @(posedge clk);
    cpu_resetn = 1;
  end

  // Watchdog: (bytes in + bytes out) * 10 bits per byte, plus slack
  initial begin
    repeat ((NumJobs + 1) * 7 * N * N * 10 * ClksPerBit + 100_000) @(posedge clk);
    $display("TIMEOUT");
    $fatal;
  end

  task automatic send_byte(input logic [7:0] d);
    uart_rx = 1'b0;  // start bit
    repeat (ClksPerBit) @(posedge clk);
    for (int i = 0; i < 8; i++) begin
      uart_rx = d[i];
      repeat (ClksPerBit) @(posedge clk);
    end
    uart_rx = 1'b1;  // stop bit
    repeat (ClksPerBit) @(posedge clk);
  endtask

  // UART monitor on the DUT's TX pin
  logic [7:0] rx_bytes[$];
  int framing_errors = 0;

  initial begin
    logic [7:0] d;
    @(posedge clk iff cpu_resetn);
    forever begin
      @(negedge uart_tx);
      repeat (ClksPerBit / 2) @(posedge clk);  // middle of start bit
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
    logic signed [31:0] y_got;

    $display("TEST N=%0d", N);

    $readmemh(AFile, a);
    $readmemh(BFile, b);
    $readmemh(CFile, c);
    $readmemh(YFile, y_exp);

    uart_rx = 1'b1;
    @(posedge clk iff cpu_resetn);
    repeat (10) @(posedge clk);

    // 1. stray bytes then silence: must not shift the following jobs
    repeat (3) send_byte(8'hA5);
    repeat (2 * IdleTimeoutClks) @(posedge clk);

    // 2. full jobs
    for (int job = 0; job < NumJobs; job++) begin
      foreach (a[i, j]) send_byte(a[i][j]);
      foreach (b[i, j]) send_byte(b[i][j]);
      foreach (c[i, j]) send_byte(c[i][j]);

      wait (rx_bytes.size() == (job + 1) * OutBytes);

      for (int i = 0; i < N; i++) begin
        for (int j = 0; j < N; j++) begin
          int base = job * OutBytes + 4 * (i * N + j);
          y_got = {rx_bytes[base+3], rx_bytes[base+2], rx_bytes[base+1], rx_bytes[base]};
          if (y_got !== y_exp[i][j]) begin
            if (errors < 10)
              $display(
                  "MISMATCH job %0d y[%0d][%0d]: got %0d, exp %0d", job, i, j, y_got, y_exp[i][j]
              );
            errors++;
          end
        end
      end
    end

    // Nothing extra should come out
    repeat (20 * ClksPerBit) @(posedge clk);
    if (rx_bytes.size() != NumJobs * OutBytes) begin
      $display("MISMATCH byte count: got %0d, exp %0d", rx_bytes.size(), NumJobs * OutBytes);
      errors++;
    end
    if (framing_errors != 0) begin
      $display("MISMATCH %0d UART framing errors", framing_errors);
      errors++;
    end
    if (led[7:4] !== 4'(NumJobs)) begin
      $display("MISMATCH job counter LEDs: got %0d, exp %0d", led[7:4], NumJobs);
      errors++;
    end

    if (errors == 0) $display("TEST PASSED N=%0d", N);
    else $display("FAIL N=%0d: %0d mismatches", N, errors);

    $finish;
  end

  initial begin
    $dumpfile(FstPath);
    $dumpvars();
  end
endmodule
