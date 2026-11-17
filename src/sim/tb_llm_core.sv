`timescale 1ns / 1ps

// llm_core through its UART pins, with axi_mem_model standing in for DDR.
//
//   1. a stray partial command, then silence (idle timeout)
//   2. STATUS
//   3. WRITE 6 random beats, READ them back
//   4. MATMUL on each vector set in VecDirs (sim/gen_qmv.py mm): the weights
//      are loaded straight into the memory model in qmv DDR format, x goes
//      over the UART, every result is checked bit-exact
//   5. SD_STORE the first matmul's weights to the card model, wipe them in
//      DDR, SD_LOAD them back, run that matmul again
//   6. CPU side over AXI4-Lite: ID, console output on the UART, and the
//      first matmul again with x written through the XQ / XS windows and
//      every result read back from the Y RAM
// The boot-time SD_LOAD (blank card: error 8) is checked through STATUS.
module tb_llm_core #(
    parameter string VecDir0,  // first matmul vectors
    parameter int N0,
    parameter int D0,
    parameter string VecDir1,  // second matmul vectors
    parameter int N1,
    parameter int D1,
    parameter int GS = 32
) ();

  localparam int ClkHz = 200_000_000;
  localparam int ClksPerBit = 8;
  localparam int IdleTimeoutClks = 40 * ClksPerBit;
  localparam int AddrW = 30;
  localparam int P = 32;
  localparam int MaxBeats = 288 * 768 / P;

  logic clk, rst_n, calib_done;
  logic uart_rx, uart_tx;
  logic [7:0] led;

  logic [AddrW-1:0] awaddr, araddr;
  logic [7:0] awlen, arlen;
  logic awvalid, awready, wlast, wvalid, wready, bvalid, bready;
  logic arvalid, arready, rvalid, rready;
  logic [255:0] wdata, rdata;
  logic sd_power_n, sd_sclk, sd_cs, sd_mosi, sd_miso;
  logic [17:0] s_awaddr, s_araddr;
  logic s_awvalid, s_awready, s_wvalid, s_wready, s_bvalid, s_bready;
  logic s_arvalid, s_arready, s_rvalid, s_rready;
  logic [31:0] s_wdata, s_rdata;

  llm_core #(
      .ClkHz(ClkHz),
      .Baud(ClkHz / ClksPerBit),
      .IdleTimeoutClks(IdleTimeoutClks),
      .GS(GS),
      .AddrW(AddrW),
      .SdSlowHalf(6),
      .SdFastHalf(3),
      .SdPowerClks(200)
  ) dut (
      .clk_i(clk),
      .rst_ni(rst_n),
      .calib_done_i(calib_done),
      .uart_rx_i(uart_rx),
      .uart_tx_o(uart_tx),
      .led_o(led),
      .m_awaddr_o(awaddr),
      .m_awlen_o(awlen),
      .m_awvalid_o(awvalid),
      .m_awready_i(awready),
      .m_wdata_o(wdata),
      .m_wlast_o(wlast),
      .m_wvalid_o(wvalid),
      .m_wready_i(wready),
      .m_bvalid_i(bvalid),
      .m_bready_o(bready),
      .m_araddr_o(araddr),
      .m_arlen_o(arlen),
      .m_arvalid_o(arvalid),
      .m_arready_i(arready),
      .m_rdata_i(rdata),
      .m_rvalid_i(rvalid),
      .m_rready_o(rready),
      .sd_power_n_o(sd_power_n),
      .sd_sclk_o(sd_sclk),
      .sd_cs_o(sd_cs),
      .sd_mosi_o(sd_mosi),
      .sd_miso_i(sd_miso),
      .sd_cd_i(1'b0),
      .s_awaddr_i(s_awaddr),
      .s_awvalid_i(s_awvalid),
      .s_awready_o(s_awready),
      .s_wdata_i(s_wdata),
      .s_wvalid_i(s_wvalid),
      .s_wready_o(s_wready),
      .s_bvalid_o(s_bvalid),
      .s_bready_i(s_bready),
      .s_araddr_i(s_araddr),
      .s_arvalid_i(s_arvalid),
      .s_arready_o(s_arready),
      .s_rdata_o(s_rdata),
      .s_rvalid_o(s_rvalid),
      .s_rready_i(s_rready)
  );

  // AXI4-Lite master. Drives on the falling edge and checks the ready a moment
  // later (#1, once the slave's combinational ready has settled); a ready
  // seen then is taken at the next rising edge. One transaction at a time.
  task automatic axil_write(input logic [17:0] a, input logic [31:0] d);
    @(negedge clk);
    s_awaddr = a;
    s_wdata = d;
    s_awvalid = 1;
    s_wvalid = 1;
    #1;
    while (!s_awready) begin
      @(negedge clk);
      #1;
    end
    @(negedge clk);  // AW and W taken at the rising edge in between
    s_awvalid = 0;
    s_wvalid = 0;
    s_bready = 1;
    #1;
    while (!s_bvalid) begin
      @(negedge clk);
      #1;
    end
    @(negedge clk);
    s_bready = 0;
  endtask

  task automatic axil_read(input logic [17:0] a, output logic [31:0] d);
    @(negedge clk);
    s_araddr = a;
    s_arvalid = 1;
    #1;
    while (!s_arready) begin
      @(negedge clk);
      #1;
    end
    @(negedge clk);
    s_arvalid = 0;
    s_rready = 1;
    #1;
    while (!s_rvalid) begin
      @(negedge clk);
      #1;
    end
    d = s_rdata;
    @(negedge clk);
    s_rready = 0;
  endtask

  sd_card_model #(
      .Blocks(512)
  ) u_card (
      .power_n(sd_power_n),
      .sclk(sd_sclk),
      .cs_n(sd_cs),
      .mosi(sd_mosi),
      .miso(sd_miso)
  );

  axi_mem_model #(
      .AddrW(AddrW)
  ) u_mem (
      .clk_i(clk),
      .rst_ni(rst_n),
      .awaddr_i(awaddr),
      .awlen_i(awlen),
      .awvalid_i(awvalid),
      .awready_o(awready),
      .wdata_i(wdata),
      .wlast_i(wlast),
      .wvalid_i(wvalid),
      .wready_o(wready),
      .bvalid_o(bvalid),
      .bready_i(bready),
      .araddr_i(araddr),
      .arlen_i(arlen),
      .arvalid_i(arvalid),
      .arready_o(arready),
      .rdata_o(rdata),
      .rvalid_o(rvalid),
      .rready_i(rready)
  );

  initial begin
    clk = 0;
    forever #2.5 clk = ~clk;
  end

  initial begin
    repeat (20_000_000) @(posedge clk);
    $display("TIMEOUT");
    $fatal;
  end

  // ---------------------------------------------------------------------------
  // UART
  // ---------------------------------------------------------------------------
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

  initial begin
    logic [7:0] d;
    @(posedge clk iff rst_n);
    forever begin
      @(negedge uart_tx);
      repeat (ClksPerBit / 2) @(posedge clk);
      for (int i = 0; i < 8; i++) begin
        repeat (ClksPerBit) @(posedge clk);
        d[i] = uart_tx;
      end
      repeat (ClksPerBit) @(posedge clk);
      rx_bytes.push_back(d);
    end
  end

  task automatic recv(input int n, output logic [7:0] out[$]);
    int t = 0;
    while (rx_bytes.size() < n) begin
      @(posedge clk);
      t++;
      if (t > 200_000 + n * 20 * ClksPerBit) begin
        $display("TIMEOUT waiting for %0d reply bytes, have %0d", n, rx_bytes.size());
        $fatal;
      end
    end
    out = {};
    repeat (n) out.push_back(rx_bytes.pop_front());
  endtask

  // ---------------------------------------------------------------------------
  // Matmul vectors
  // ---------------------------------------------------------------------------
  logic [P*8-1:0] xq[2048/P];
  logic [31:0] xs[2048/GS];
  logic [P*8+31:0] w[MaxBeats];
  logic [31:0] y_exp[1024];

  int errors = 0;

  // STATUS: returns the flags byte, error code and image size
  task automatic status(output logic [7:0] flags, output logic [7:0] code,
                        output logic [31:0] img_bytes);
    logic [7:0] r[$];
    send_byte(8'h00);
    recv(12, r);
    if (r[0] != "L" || r[1] != "L" || r[2] != "M") begin
      $display("MISMATCH status reply %h %h %h", r[0], r[1], r[2]);
      errors++;
    end
    flags = r[3];
    code = r[4];
    img_bytes = {r[11], r[10], r[9], r[8]};
  endtask

  // wait for the SD operation to finish, then check its result
  task automatic sd_wait(input string what, input bit exp_ok, input logic [7:0] exp_code);
    logic [7:0] flags, code;
    logic [31:0] nb;
    do begin
      repeat (2000) @(posedge clk);
      status(flags, code, nb);
    end while (flags[1]);
    if (flags[2] !== exp_ok || (!exp_ok && code !== exp_code)) begin
      $display("MISMATCH %s: flags %b code %0d", what, flags, code);
      errors++;
    end else begin
      $display("%s: %s", what, exp_ok ? "ok" : $sformatf("error %0d as expected", code));
    end
  endtask

  task automatic run_matmul(input string dir, input int n, input int d, input int base_word);
    logic [7:0] reply[$];
    int beats = d * n / P;
    logic [31:0] y_got;
    int errs = 0;

    $readmemh({dir, "/xq.hex"}, xq);
    $readmemh({dir, "/xs.hex"}, xs);
    $readmemh({dir, "/w.hex"}, w);
    $readmemh({dir, "/y.hex"}, y_exp);

    // qmv DDR format: per 8 beats, one beat of 8 scales then the 8 weight beats
    for (int c = 0; c < beats / 8; c++) begin
      logic [255:0] sbeat;
      for (int j = 0; j < 8; j++) begin
        sbeat[j*32+:32] = w[c*8+j][P*8+:32];
        u_mem.mem[base_word+c*9+1+j] = w[c*8+j][P*8-1:0];
      end
      u_mem.mem[base_word+c*9] = sbeat;
    end

    send_byte(8'h03);
    send_u32(32'(base_word * 32));
    send_u32({16'(d), 16'(n)});
    for (int i = 0; i < n / P; i++) for (int b = 0; b < P; b++) send_byte(xq[i][b*8+:8]);
    for (int i = 0; i < n / GS; i++) send_u32(xs[i]);

    recv(4 * d, reply);
    for (int i = 0; i < d; i++) begin
      y_got = {reply[4*i+3], reply[4*i+2], reply[4*i+1], reply[4*i]};
      if (y_got !== y_exp[i]) begin
        if (errs < 5) $display("MISMATCH %s y[%0d]: got %08h, exp %08h", dir, i, y_got, y_exp[i]);
        errs++;
      end
    end
    $display("matmul %0dx%0d: %0d mismatches", d, n, errs);
    errors += errs;
  endtask

  // ---------------------------------------------------------------------------
  // Sequence
  // ---------------------------------------------------------------------------
  initial begin
    logic [7:0] reply[$];
    logic [255:0] beats[6];

    $display("TEST llm_core");
    uart_rx = 1'b1;
    s_awvalid = 0;
    s_wvalid = 0;
    s_bready = 0;
    s_arvalid = 0;
    s_rready = 0;
    rst_n = 0;
    calib_done = 0;
    repeat (10) @(posedge clk);
    rst_n = 1;
    repeat (100) @(posedge clk);
    calib_done = 1;

    // 1. stray partial command
    // (while the boot SD_LOAD runs the opcode is dropped, so the second byte
    // must not be a valid opcode either)
    send_byte(8'h01);
    send_byte(8'hAB);
    repeat (2 * IdleTimeoutClks) @(posedge clk);

    // 2. STATUS; the boot-time SD_LOAD finds a blank card
    sd_wait("boot SD_LOAD, blank card", 0, 8'd8);

    // 3. WRITE, READ back
    foreach (beats[i]) beats[i] = {8{$urandom()}};
    send_byte(8'h01);
    send_u32(32'h0001_0040);
    send_u32(6);
    foreach (beats[i]) for (int b = 0; b < 32; b++) send_byte(beats[i][b*8+:8]);
    recv(1, reply);
    if (reply[0] != 8'h01) begin
      $display("MISMATCH write ack %h", reply[0]);
      errors++;
    end
    send_byte(8'h02);
    send_u32(32'h0001_0040);
    send_u32(6);
    recv(6 * 32, reply);
    foreach (beats[i]) begin
      for (int b = 0; b < 32; b++) begin
        if (reply[i*32+b] != beats[i][b*8+:8]) begin
          if (errors < 10) $display("MISMATCH read beat %0d byte %0d", i, b);
          errors++;
        end
      end
    end

    // 4. MATMUL
    run_matmul(VecDir0, N0, D0, 32'h0002_0000 / 32);
    run_matmul(VecDir1, N1, D1, 32'h0004_0000 / 32);

    // 5. SD round trip of the first matmul's weights
    begin
      int words = (D0 * N0 / P) / 8 * 9;
      logic [7:0] flags, code;
      logic [31:0] nb;
      send_byte(8'h05);
      send_u32(32'h0002_0000);
      send_u32(32'(words * 32));
      recv(1, reply);
      if (reply[0] != 8'h05) begin
        $display("MISMATCH SD_STORE ack %h", reply[0]);
        errors++;
      end
      sd_wait("SD_STORE", 1, 0);
      for (int i = 0; i < words; i++) u_mem.mem[32'h0002_0000/32+i] = '0;
      send_byte(8'h04);
      recv(1, reply);
      if (reply[0] != 8'h04) begin
        $display("MISMATCH SD_LOAD ack %h", reply[0]);
        errors++;
      end
      sd_wait("SD_LOAD", 1, 0);
      status(flags, code, nb);
      if (nb != words * 32) begin
        $display("MISMATCH loaded image size %0d, exp %0d", nb, words * 32);
        errors++;
      end
      // run_matmul would reload the weights; send just the command
      send_byte(8'h03);
      send_u32(32'h0002_0000);
      send_u32({16'(D0), 16'(N0)});
      $readmemh({VecDir0, "/xq.hex"}, xq);
      $readmemh({VecDir0, "/xs.hex"}, xs);
      $readmemh({VecDir0, "/y.hex"}, y_exp);
      for (int i = 0; i < N0 / P; i++) for (int b = 0; b < P; b++) send_byte(xq[i][b*8+:8]);
      for (int i = 0; i < N0 / GS; i++) send_u32(xs[i]);
      recv(4 * D0, reply);
      begin
        int errs = 0;
        for (int i = 0; i < D0; i++) begin
          if ({reply[4*i+3], reply[4*i+2], reply[4*i+1], reply[4*i]} !== y_exp[i]) errs++;
        end
        $display("matmul from SD-loaded weights: %0d mismatches", errs);
        errors += errs;
      end
    end

    // 6. CPU side
    begin
      logic [31:0] v;
      string msg = "Hi from the CPU\n";
      int errs = 0;
      axil_read(18'h00000, v);
      if (v !== 32'h314D_4C4C) begin
        $display("MISMATCH ID register %h", v);
        errors++;
      end
      foreach (msg[i]) axil_write(18'h00008, 32'(msg[i]));
      recv(msg.len(), reply);
      foreach (reply[i]) begin
        if (reply[i] != msg[i]) begin
          $display("MISMATCH console byte %0d: %h, exp %h", i, reply[i], msg[i]);
          errors++;
        end
      end

      // x for the first matmul, through the CPU windows
      for (int i = 0; i < N0 / P; i++)
        for (int w = 0; w < 8; w++) axil_write(18'h01000 + 18'((i * 8 + w) * 4), xq[i][w*32+:32]);
      for (int g = 0; g < N0 / GS; g++) axil_write(18'h01800 + 18'(g * 4), xs[g]);
      axil_write(18'h00010, 32'h0002_0000);
      axil_write(18'h00014, {16'(D0), 16'(N0)});
      axil_write(18'h00018, 32'd1);
      do axil_read(18'h00004, v); while (v[8]);
      for (int i = 0; i < D0; i++) begin
        axil_read(18'h20000 + 18'(i * 4), v);
        if (v !== y_exp[i]) begin
          if (errs < 5) $display("MISMATCH CPU matmul y[%0d]: got %08h, exp %08h", i, v, y_exp[i]);
          errs++;
        end
      end
      $display("CPU matmul %0dx%0d through the registers: %0d mismatches", D0, N0, errs);
      errors += errs;
    end

    repeat (100) @(posedge clk);
    if (rx_bytes.size() != 0) begin
      $display("MISMATCH %0d extra reply bytes", rx_bytes.size());
      errors++;
    end
    if (u_mem.errors != 0) begin
      $display("MISMATCH %0d AXI protocol errors", u_mem.errors);
      errors++;
    end
    if (u_card.errors != 0) begin
      $display("MISMATCH %0d SD protocol errors", u_card.errors);
      errors++;
    end
    if (led[7:4] !== 4'd4) begin
      $display("MISMATCH job LEDs %0d, exp 4", led[7:4]);
      errors++;
    end

    if (errors == 0) $display("TEST PASSED");
    else $display("FAIL: %0d mismatches", errors);
    $finish;
  end
endmodule
