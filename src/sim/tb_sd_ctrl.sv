`timescale 1ns / 1ps

// sd_ctrl against sd_card_model and axi_mem_model:
//   1. LOAD from a blank card: bad header (error 8)
//   2. STORE a random image (not a whole number of blocks) from DDR to the card
//   3. wipe that DDR region, LOAD it back, compare with the original
//   4. flip a byte on the card, LOAD: checksum mismatch (error 9)
module tb_sd_ctrl #(
    parameter bit Sdhc = 1
) ();

  localparam int AddrW = 30;
  localparam int ImgAddr = 32'h0000_2000;
  localparam int ImgBytes = 1600;  // 3 blocks + 64 bytes
  localparam int ImgWords = ImgBytes / 32;

  logic clk, rst_n;
  logic load, store;
  logic busy, ok, err;
  logic [3:0] err_code;
  logic [31:0] img_bytes, img_addr;
  logic sd_power_n, sd_sclk, sd_cs, sd_mosi, sd_miso;

  logic [AddrW-1:0] awaddr, araddr;
  logic awvalid, awready, wvalid, wready, bvalid, bready;
  logic arvalid, arready, rvalid, rready;
  logic [255:0] wdata, rdata;

  sd_ctrl #(
      .ClkHz(200_000_000),
      .SlowHalf(6),
      .FastHalf(3),
      .PowerClks(200),
      .AddrW(AddrW)
  ) dut (
      .clk_i(clk),
      .rst_ni(rst_n),
      .load_i(load),
      .store_i(store),
      .store_addr_i(ImgAddr),
      .store_bytes_i(ImgBytes),
      .busy_o(busy),
      .ok_o(ok),
      .err_o(err),
      .err_code_o(err_code),
      .img_bytes_o(img_bytes),
      .img_addr_o(img_addr),
      .sd_power_n_o(sd_power_n),
      .sd_sclk_o(sd_sclk),
      .sd_cs_o(sd_cs),
      .sd_mosi_o(sd_mosi),
      .sd_miso_i(sd_miso),
      .m_awaddr_o(awaddr),
      .m_awvalid_o(awvalid),
      .m_awready_i(awready),
      .m_wdata_o(wdata),
      .m_wvalid_o(wvalid),
      .m_wready_i(wready),
      .m_bvalid_i(bvalid),
      .m_bready_o(bready),
      .m_araddr_o(araddr),
      .m_arvalid_o(arvalid),
      .m_arready_i(arready),
      .m_rdata_i(rdata),
      .m_rvalid_i(rvalid),
      .m_rready_o(rready)
  );

  sd_card_model #(
      .Sdhc(Sdhc)
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
      .awlen_i(8'd0),
      .awvalid_i(awvalid),
      .awready_o(awready),
      .wdata_i(wdata),
      .wstrb_i('1),
      .wlast_i(1'b1),
      .wvalid_i(wvalid),
      .wready_o(wready),
      .bvalid_o(bvalid),
      .bready_i(bready),
      .araddr_i(araddr),
      .arlen_i(8'd0),
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

  int errors = 0;

  task automatic run(input bit do_store, input string what, input bit exp_ok,
                     input logic [3:0] exp_code);
    @(negedge clk);
    if (do_store) store = 1;
    else load = 1;
    @(negedge clk);
    store = 0;
    load = 0;
    wait (!busy);
    @(negedge clk);
    if (ok !== exp_ok || (!exp_ok && err_code !== exp_code)) begin
      $display("MISMATCH %s: ok=%b err=%b code=%0d, exp ok=%b code=%0d", what, ok, err, err_code,
               exp_ok, exp_code);
      errors++;
    end else begin
      $display("%s: %s", what, ok ? "ok" : $sformatf("error %0d as expected", err_code));
    end
  endtask

  logic [255:0] orig[ImgWords];

  initial begin
    $display("TEST sd_ctrl Sdhc=%0d", Sdhc);
    rst_n = 0;
    load = 0;
    store = 0;
    foreach (u_card.mem[i]) u_card.mem[i] = 8'h00;
    repeat (10) @(posedge clk);
    rst_n = 1;
    repeat (10) @(posedge clk);

    // 1. blank card
    run(0, "LOAD blank card", 0, 4'd8);

    // 2. STORE
    for (int i = 0; i < ImgWords; i++) begin
      orig[i] = {8{$urandom()}};
      u_mem.mem[ImgAddr/32+i] = orig[i];
    end
    run(1, "STORE", 1, 0);

    // 3. LOAD back into a wiped region
    for (int i = 0; i < ImgWords; i++) u_mem.mem[ImgAddr/32+i] = '0;
    u_mem.mem[ImgAddr/32+ImgWords] = {8{32'hA5A5_5A5A}};  // must survive
    run(0, "LOAD", 1, 0);
    if (img_bytes != ImgBytes || img_addr != ImgAddr) begin
      $display("MISMATCH header: %0d bytes at %h", img_bytes, img_addr);
      errors++;
    end
    for (int i = 0; i < ImgWords; i++) begin
      if (u_mem.mem[ImgAddr/32+i] !== orig[i]) begin
        if (errors < 10) $display("MISMATCH DDR beat %0d after LOAD", i);
        errors++;
      end
    end
    // nothing written past the image
    if (u_mem.mem[ImgAddr/32+ImgWords] !== {8{32'hA5A5_5A5A}}) begin
      $display("MISMATCH LOAD wrote past the image");
      errors++;
    end

    // 4. corrupted card
    u_card.mem[512 + 700] = ~u_card.mem[512 + 700];
    run(0, "LOAD corrupted card", 0, 4'd9);

    if (u_card.errors != 0) begin
      $display("MISMATCH %0d SD protocol errors", u_card.errors);
      errors++;
    end
    if (u_mem.errors != 0) begin
      $display("MISMATCH %0d AXI protocol errors", u_mem.errors);
      errors++;
    end
    if (errors == 0) $display("TEST PASSED");
    else $display("FAIL: %0d mismatches", errors);
    $finish;
  end
endmodule
