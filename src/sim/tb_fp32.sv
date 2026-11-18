`timescale 1ns / 1ps

// Checks fp32_add, fp32_mul, int_to_fp32, fp32_div and fp32_exp against
// numpy float32 results (sim/gen_qmv.py fp; exp: the firmware's m_expf). Vectors go in back to back with random gaps; the tag
// carries the vector index so each result is checked against its own vector.
module tb_fp32 #(
    parameter string VecDir,
    parameter int NumAdd,  // vectors in add.hex, mul.hex, i2f.hex
    parameter int NumMul,
    parameter int NumI2f,
    parameter int NumDiv,
    parameter int NumExp,
    parameter int MaxVec = 32768
) ();

  logic clk, rst_n;
  initial begin
    clk = 0;
    forever #2.5 clk = ~clk;
  end

  logic [31:0] add_v[3*MaxVec];
  logic [31:0] mul_v[3*MaxVec];
  logic [31:0] i2f_v[2*MaxVec];
  logic [31:0] div_v[3*MaxVec];
  logic [31:0] exp_v[2*MaxVec];


  logic in_valid;
  logic [15:0] in_tag;
  logic [31:0] add_a, add_b, mul_a, mul_b, i2f_a, div_a, div_b, exp_x;
  logic add_ov, mul_ov, i2f_ov, div_ov, exp_ov;
  logic [15:0] add_ot, mul_ot, i2f_ot, div_ot, exp_ot;
  logic [31:0] add_y, mul_y, i2f_y, div_y, exp_y;
  int unit;  // 0 add, 1 mul, 2 i2f, 3 div, 4 exp: one unit at a time

  fp32_add #(.TagW(16)) u_add (
      .clk_i(clk), .rst_ni(rst_n), .valid_i(in_valid && unit == 0), .tag_i(in_tag),
      .a_i(add_a), .b_i(add_b), .valid_o(add_ov), .tag_o(add_ot), .y_o(add_y)
  );
  fp32_mul #(.TagW(16)) u_mul (
      .clk_i(clk), .rst_ni(rst_n), .valid_i(in_valid && unit == 1), .tag_i(in_tag),
      .a_i(mul_a), .b_i(mul_b), .valid_o(mul_ov), .tag_o(mul_ot), .y_o(mul_y)
  );
  int_to_fp32 #(.TagW(16)) u_i2f (
      .clk_i(clk), .rst_ni(rst_n), .valid_i(in_valid && unit == 2), .tag_i(in_tag),
      .a_i(i2f_a), .valid_o(i2f_ov), .tag_o(i2f_ot), .y_o(i2f_y)
  );
  fp32_div #(.TagW(16)) u_div (
      .clk_i(clk), .rst_ni(rst_n), .valid_i(in_valid && unit == 3), .tag_i(in_tag),
      .a_i(div_a), .b_i(div_b), .valid_o(div_ov), .tag_o(div_ot), .y_o(div_y)
  );
  fp32_exp #(.TagW(16)) u_exp (
      .clk_i(clk), .rst_ni(rst_n), .valid_i(in_valid && unit == 4), .tag_i(in_tag),
      .x_i(exp_x), .valid_o(exp_ov), .tag_o(exp_ot), .y_o(exp_y)
  );

  int errors = 0;
  int checked[5] = '{0, 0, 0, 0, 0};

  always @(posedge clk iff rst_n) begin
    if (add_ov) begin
      checked[0]++;
      if (add_y !== add_v[3*add_ot+2]) begin
        if (errors < 10)
          $display("MISMATCH add %08h + %08h: got %08h, exp %08h", add_v[3*add_ot],
                   add_v[3*add_ot+1], add_y, add_v[3*add_ot+2]);
        errors++;
      end
    end
    if (mul_ov) begin
      checked[1]++;
      if (mul_y !== mul_v[3*mul_ot+2]) begin
        if (errors < 10)
          $display("MISMATCH mul %08h * %08h: got %08h, exp %08h", mul_v[3*mul_ot],
                   mul_v[3*mul_ot+1], mul_y, mul_v[3*mul_ot+2]);
        errors++;
      end
    end
    if (div_ov) begin
      checked[3]++;
      if (div_y !== div_v[3*div_ot+2]) begin
        if (errors < 10)
          $display("MISMATCH div %08h / %08h: got %08h, exp %08h", div_v[3*div_ot],
                   div_v[3*div_ot+1], div_y, div_v[3*div_ot+2]);
        errors++;
      end
    end
    if (exp_ov) begin
      checked[4]++;
      if (exp_y !== exp_v[2*exp_ot+1]) begin
        if (errors < 10)
          $display("MISMATCH exp %08h: got %08h, exp %08h", exp_v[2*exp_ot], exp_y,
                   exp_v[2*exp_ot+1]);
        errors++;
      end
    end
    if (i2f_ov) begin
      checked[2]++;
      if (i2f_y !== i2f_v[2*i2f_ot+1]) begin
        if (errors < 10)
          $display("MISMATCH i2f %08h: got %08h, exp %08h", i2f_v[2*i2f_ot], i2f_y,
                   i2f_v[2*i2f_ot+1]);
        errors++;
      end
    end
  end

  initial begin
    int n[5];
    $display("TEST fp32");
    $readmemh({VecDir, "/add.hex"}, add_v);
    $readmemh({VecDir, "/mul.hex"}, mul_v);
    $readmemh({VecDir, "/i2f.hex"}, i2f_v);
    $readmemh({VecDir, "/div.hex"}, div_v);
    $readmemh({VecDir, "/exp.hex"}, exp_v);
    n = '{NumAdd, NumMul, NumI2f, NumDiv, NumExp};

    rst_n = 0;
    in_valid = 0;
    unit = 0;
    repeat (5) @(posedge clk);
    rst_n <= 1;
    @(posedge clk);

    for (unit = 0; unit < 5; unit++) begin
      for (int i = 0; i < n[unit]; i++) begin
        // random idle cycles between some vectors
        while ($urandom_range(0, 3) == 0) begin
          in_valid <= 0;
          @(posedge clk);
        end
        in_valid <= 1;
        in_tag <= 16'(i);
        add_a <= add_v[3*i];
        add_b <= add_v[3*i+1];
        mul_a <= mul_v[3*i];
        mul_b <= mul_v[3*i+1];
        i2f_a <= i2f_v[2*i];
        div_a <= div_v[3*i];
        div_b <= div_v[3*i+1];
        exp_x <= exp_v[2*i];
        @(posedge clk);
      end
      in_valid <= 0;
      repeat (200) @(posedge clk);
    end

    for (int u = 0; u < 5; u++) begin
      if (checked[u] != n[u]) begin
        $display("MISMATCH unit %0d: checked %0d of %0d vectors", u, checked[u], n[u]);
        errors++;
      end
    end
    $display("checked %0d add, %0d mul, %0d i2f, %0d div, %0d exp", checked[0], checked[1],
             checked[2], checked[3], checked[4]);
    if (errors == 0) $display("TEST PASSED");
    else $display("FAIL: %0d mismatches", errors);
    $finish;
  end
endmodule
