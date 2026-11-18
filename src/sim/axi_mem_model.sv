`timescale 1ns / 1ps

// Behavioral AXI4 slave memory standing in for the DDR controller in
// simulation. 256-bit data, INCR bursts. Every ready / valid has random
// stalls and reads have a random latency, so the master sees realistic
// back-pressure. Multiple outstanding reads are served in order.
//
// mem is one 256-bit word per 32 bytes; testbenches can load it directly.
// Writes honor the byte strobes.
module axi_mem_model #(
    parameter int AddrW = 30,
    parameter int Words = 1 << 16  // 2 MB
) (
    input wire logic clk_i,
    input wire logic rst_ni,
    input wire logic [AddrW-1:0] awaddr_i,
    input wire logic [7:0] awlen_i,
    input wire logic awvalid_i,
    output logic awready_o,
    input wire logic [255:0] wdata_i,
    input wire logic [31:0] wstrb_i,
    input wire logic wlast_i,
    input wire logic wvalid_i,
    output logic wready_o,
    output logic bvalid_o,
    input wire logic bready_i,
    input wire logic [AddrW-1:0] araddr_i,
    input wire logic [7:0] arlen_i,
    input wire logic arvalid_i,
    output logic arready_o,
    output logic [255:0] rdata_o,
    output logic rvalid_o,
    input wire logic rready_i
);

  logic [255:0] mem[Words];

  int ar_addr_q[$];
  int ar_len_q[$];
  int aw_addr_q[$];
  int b_pending = 0;
  int errors = 0;

  function automatic int word(input logic [AddrW-1:0] addr);
    if (addr[4:0] != 0) begin
      $display("AXI ERROR: unaligned address %h", addr);
      errors++;
    end
    return int'(addr >> 5) % Words;
  endfunction

  // ---- read address
  always @(posedge clk_i) begin
    if (!rst_ni) arready_o <= 0;
    else begin
      if (arvalid_i && arready_o) begin
        if ((araddr_i & 30'hFFF) + (arlen_i + 1) * 32 > 4096) begin
          $display("AXI ERROR: read burst at %h len %0d crosses 4 KB", araddr_i, arlen_i + 1);
          errors++;
        end
        ar_addr_q.push_back(word(araddr_i));
        ar_len_q.push_back(arlen_i + 1);
      end
      arready_o <= ($urandom_range(0, 3) != 0);
    end
  end

  // ---- read data: in order, random latency and gaps. Sampled at the clock
  // edge like the DUT's flops (no "@(posedge iff ...)" waits, which race with
  // the DUT dropping rready on the handshake edge).
  int r_addr = 0, r_left = 0, r_wait = 0;
  always @(posedge clk_i) begin
    if (!rst_ni) begin
      rvalid_o <= 0;
      r_left = 0;
    end else begin
      if (rvalid_o && rready_i) begin
        r_addr++;
        r_left--;
      end
      if (r_left == 0 && ar_addr_q.size() != 0) begin
        r_addr = ar_addr_q.pop_front();
        r_left = ar_len_q.pop_front();
        r_wait = $urandom_range(5, 30);
      end
      if (r_wait > 0) r_wait--;

      if (rvalid_o && !rready_i) begin
        // hold the beat until it is taken
      end else if (r_left > 0 && r_wait == 0 && $urandom_range(0, 7) != 0) begin
        rvalid_o <= 1;
        rdata_o <= mem[r_addr%Words];
      end else begin
        rvalid_o <= 0;
      end
    end
  end

  // ---- write address / data / response
  int w_beat = 0;
  always @(posedge clk_i) begin
    if (!rst_ni) begin
      awready_o <= 0;
      wready_o <= 0;
      bvalid_o <= 0;
    end else begin
      if (awvalid_i && awready_o) aw_addr_q.push_back(word(awaddr_i));
      if (wvalid_i && wready_o) begin
        if (aw_addr_q.size() == 0) begin
          // data before its address: hold it until AW arrives (not expected
          // from llm_core, which waits for both; flag it)
          $display("AXI ERROR: W beat before AW");
          errors++;
        end else begin
          for (int b = 0; b < 32; b++)
            if (wstrb_i[b]) mem[(aw_addr_q[0]+w_beat)%Words][8*b+:8] <= wdata_i[8*b+:8];
          w_beat++;
          if (wlast_i) begin
            void'(aw_addr_q.pop_front());
            w_beat = 0;
            b_pending++;
          end
        end
      end
      if (bvalid_o && bready_i) b_pending--;
      // once raised, bvalid stays up until taken
      bvalid_o <= (bvalid_o && !bready_i) || (b_pending > 0 && $urandom_range(0, 2) != 0);
      awready_o <= ($urandom_range(0, 2) != 0);
      wready_o <= (aw_addr_q.size() != 0 || (awvalid_i && awready_o)) && ($urandom_range(0, 2) != 0);
    end
  end

endmodule
