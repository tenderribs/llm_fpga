`timescale 1ns / 1ps
`default_nettype none

// attn.sv against sim/gen_attn.py vectors: loads the A RAM, streams the beats
// with random gaps, checks every result bit for bit (sim/run_attn.sh).
module tb_attn #(
    parameter string VecDir = "",
    parameter int Mode = 0,
    parameter int HS = 48,
    parameter int NT = 1,
    parameter int NBeats = 1,
    parameter int NOut = 1
);
  logic clk = 0, rst_n = 0;
  always #2 clk = ~clk;

  logic start, busy, done, a_we, b_valid, b_ready, y_valid;
  logic [11:0] a_addr;
  logic [31:0] a_data, y;
  logic [255:0] b_data;
  logic [15:0] y_idx;

  attn dut (
      .clk_i(clk),
      .rst_ni(rst_n),
      .start_i(start),
      .mode_i(1'(Mode)),
      .hs_i(9'(HS)),
      .nt_i(16'(NT)),
      .beats_i(32'(NBeats)),
      .busy_o(busy),
      .done_o(done),
      .a_we_i(a_we),
      .a_addr_i(a_addr),
      .a_data_i(a_data),
      .b_valid_i(b_valid),
      .b_ready_o(b_ready),
      .b_data_i(b_data),
      .y_valid_o(y_valid),
      .y_idx_o(y_idx),
      .y_o(y)
  );

  logic [31:0] a_mem[4096];
  logic [255:0] beats[NBeats];
  logic [31:0] y_exp[NOut];
  logic [31:0] y_got[NOut];
  bit seen[NOut];
  int errors = 0, nres = 0;

  always @(posedge clk) begin
    if (rst_n && y_valid) begin
      if (y_idx >= NOut) begin
        $display("MISMATCH result index %0d out of range", y_idx);
        errors++;
      end else begin
        if (seen[y_idx]) begin
          $display("MISMATCH result %0d twice", y_idx);
          errors++;
        end
        seen[y_idx] = 1;
        y_got[y_idx] = y;
        nres++;
      end
    end
  end

  int bi;
  initial begin
    $readmemh({VecDir, "/a.hex"}, a_mem);
    $readmemh({VecDir, "/beats.hex"}, beats);
    $readmemh({VecDir, "/y.hex"}, y_exp);
    start = 0;
    a_we = 0;
    b_valid = 0;
    repeat (5) @(posedge clk);
    rst_n = 1;
    @(negedge clk);
    for (int i = 0; i < (Mode ? NT : HS); i++) begin
      a_we = 1;
      a_addr = 12'(i);
      a_data = a_mem[i];
      @(negedge clk);
    end
    a_we = 0;
    start = 1;
    @(negedge clk);
    start = 0;
    // beats with random gaps (like the DDR reader's FIFO)
    bi = 0;
    while (bi < NBeats) begin
      b_valid = ($urandom % 4) != 0;
      b_data = beats[bi];
      @(posedge clk);
      if (b_valid && b_ready) bi++;
      @(negedge clk);
    end
    b_valid = 0;
    for (int c = 0; !done; c++) begin
      if (c == 100000) begin
        $display("TIMEOUT");
        $finish;
      end
      @(posedge clk);
    end
    repeat (5) @(posedge clk);
    if (busy) begin
      $display("MISMATCH still busy after done");
      errors++;
    end
    if (nres != NOut) begin
      $display("MISMATCH %0d results, exp %0d", nres, NOut);
      errors++;
    end
    for (int i = 0; i < NOut; i++) begin
      if (y_got[i] !== y_exp[i]) begin
        if (errors < 8) $display("MISMATCH y[%0d]: got %08h, exp %08h", i, y_got[i], y_exp[i]);
        errors++;
      end
    end
    $display("TEST attn mode %0d hs %0d nt %0d", Mode, HS, NT);
    if (errors == 0) $display("TEST PASSED");
    else $display("FAIL: %0d mismatches", errors);
    $finish;
  end
endmodule

`default_nettype wire
