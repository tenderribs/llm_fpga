`timescale 1ns / 1ps

// Behavioral microSD card, SPI mode. Enough of the protocol for sd_ctrl:
// CMD0, CMD8, CMD55 + ACMD41 (ready after a few tries), CMD58, CMD16, CMD17,
// CMD18 (blocks until deselected or powered off), CMD24, CMD25 (with the
// 0xFD stop token). Responses come after a random 1..8 byte delay, read
// data after a random token delay, writes are followed by a random busy
// period. mem keeps its contents across power cycles, like a real card.
//
// errors counts protocol violations it notices (bad CRC on CMD0 / CMD8,
// commands before init, data before a write command).
module sd_card_model #(
    parameter bit Sdhc = 1,  // block addressed (SDHC) or byte addressed (SDSC)
    parameter int Blocks = 1024
) (
    input wire logic power_n,
    input wire logic sclk,
    input wire logic cs_n,
    input wire logic mosi,
    output logic miso
);

  logic [7:0] mem[Blocks*512];
  int errors = 0;

  // ---- byte-level SPI, mode 0
  logic [7:0] rx_sh, tx_sh;
  int bitc = 0;
  // queued reply bytes; bit 8 marks the last byte of a data block
  logic [8:0] out_q[$];

  assign miso = (cs_n || power_n) ? 1'b1 : tx_sh[7];

  // ---- card state
  logic idle, app_cmd, initialized;
  int acmd41_tries;
  logic [7:0] cmd[6];
  int cmd_len;
  // write state
  int wmode;  // 0 none, 1 single (CMD24), 2 multi (CMD25)
  int wcount;  // bytes of the current block received, -1 = waiting for token
  int waddr;
  logic [7:0] wblk[514];
  // multi-block read
  logic reading;
  int raddr;
  // clocks the card still needs after a data block before it takes a command
  int settle = 0;

  task automatic reset_card();
    idle = 1'b0;
    app_cmd = 1'b0;
    initialized = 1'b0;
    acmd41_tries = 0;
    cmd_len = 0;
    wmode = 0;
    wcount = -1;
    reading = 1'b0;
    out_q = {};
  endtask

  initial reset_card();

  task automatic push_ncr();
    repeat ($urandom_range(1, 8)) out_q.push_back(8'hFF);
  endtask

  task automatic push_block(input int byte_addr);
    repeat ($urandom_range(1, 40)) out_q.push_back(8'hFF);
    out_q.push_back(8'hFE);
    for (int i = 0; i < 512; i++) out_q.push_back(mem[(byte_addr+i)%(Blocks*512)]);
    out_q.push_back(8'h12);  // CRC, not checked in SPI mode
    out_q.push_back({1'b1, 8'h34});  // last byte of the block
  endtask

  function automatic int addr_of(input logic [31:0] arg);
    return Sdhc ? int'(arg) * 512 : int'(arg);
  endfunction

  task automatic handle_cmd();
    logic [5:0] idx = cmd[0][5:0];
    logic [31:0] arg = {cmd[1], cmd[2], cmd[3], cmd[4]};
    logic acmd = app_cmd;
    app_cmd = 1'b0;
    push_ncr();
    if (idx == 0) begin
      if (cmd[5] != 8'h95) begin
        $display("SD MODEL: CMD0 with bad CRC %h", cmd[5]);
        errors++;
      end
      idle = 1'b1;
      initialized = 1'b0;
      out_q.push_back(8'h01);
    end else if (!idle && !initialized) begin
      $display("SD MODEL: CMD%0d before CMD0", idx);
      errors++;
    end else if (idx == 8) begin
      if (cmd[5] != 8'h87) begin
        $display("SD MODEL: CMD8 with bad CRC %h", cmd[5]);
        errors++;
      end
      out_q.push_back(8'h01);
      out_q.push_back(8'h00);
      out_q.push_back(8'h00);
      out_q.push_back({4'h0, arg[11:8]});
      out_q.push_back(arg[7:0]);
    end else if (idx == 55) begin
      app_cmd = 1'b1;
      out_q.push_back(idle ? 8'h01 : 8'h00);
    end else if (idx == 41 && acmd) begin
      acmd41_tries++;
      if (acmd41_tries >= 3) begin
        idle = 1'b0;
        initialized = 1'b1;
      end
      out_q.push_back(idle ? 8'h01 : 8'h00);
    end else if (!initialized) begin
      $display("SD MODEL: CMD%0d before init finished", idx);
      errors++;
      out_q.push_back(8'h05);
    end else if (idx == 58) begin
      out_q.push_back(8'h00);
      out_q.push_back(Sdhc ? 8'hC0 : 8'h80);
      out_q.push_back(8'hFF);
      out_q.push_back(8'h80);
      out_q.push_back(8'h00);
    end else if (idx == 16) begin
      if (arg != 512) begin
        $display("SD MODEL: CMD16 block length %0d", arg);
        errors++;
      end
      out_q.push_back(8'h00);
    end else if (idx == 17) begin
      out_q.push_back(8'h00);
      push_block(addr_of(arg));
    end else if (idx == 18) begin
      out_q.push_back(8'h00);
      reading = 1'b1;
      raddr = addr_of(arg);
      push_block(raddr);
      raddr += 512;
    end else if (idx == 24 || idx == 25) begin
      out_q.push_back(8'h00);
      wmode = (idx == 24) ? 1 : 2;
      wcount = -1;
      waddr = addr_of(arg);
    end else begin
      $display("SD MODEL: unsupported CMD%0d", idx);
      errors++;
      out_q.push_back(8'h04);
    end
  endtask

  task automatic got_byte(input logic [7:0] b);
    if (wmode != 0) begin
      if (wcount < 0) begin
        if ((wmode == 1 && b == 8'hFE) || (wmode == 2 && b == 8'hFC)) begin
          wcount = 0;
        end else if (wmode == 2 && b == 8'hFD) begin
          wmode = 0;  // stop token: one byte, then busy
          out_q.push_back(8'hFF);
          repeat ($urandom_range(1, 30)) out_q.push_back(8'h00);
        end else if (b != 8'hFF) begin
          $display("SD MODEL: unexpected byte %h while waiting for a data token", b);
          errors++;
        end
      end else begin
        wblk[wcount] = b;
        wcount++;
        if (wcount == 514) begin
          for (int i = 0; i < 512; i++) mem[(waddr+i)%(Blocks*512)] = wblk[i];
          waddr += 512;
          wcount = -1;
          out_q.push_back(8'hE5);  // data accepted (xxx0_0101)
          repeat ($urandom_range(1, 60)) out_q.push_back(8'h00);  // busy
          if (wmode == 1) wmode = 0;
        end
      end
    end else if (cmd_len > 0 || (b[7:6] == 2'b01 && out_q.size() == 0) || (b[7:6] == 2'b01 && reading)) begin
      if (cmd_len == 0) reading = 1'b0;
      cmd[cmd_len] = b;
      cmd_len++;
      if (cmd_len == 6) begin
        cmd_len = 0;
        out_q = {};
        handle_cmd();
      end
    end
  endtask

  always @(posedge power_n) reset_card();

  always @(negedge cs_n) begin
    if (!power_n && settle > 0) begin
      $display("SD MODEL: selected again only %0d clocks after a data block (needs 8)",
               settle > 8 ? 0 : 8 - settle);
      errors++;
    end
    if (!power_n) begin
      bitc = 0;
      tx_sh = (out_q.size() != 0) ? out_q[0][7:0] : 8'hFF;
      if (out_q.size() != 0) void'(out_q.pop_front());
    end
  end

  always @(posedge cs_n) cmd_len = 0;

  always @(posedge sclk) if (settle > 0) settle--;

  always @(posedge sclk) begin
    if (!cs_n && !power_n) begin
      rx_sh = {rx_sh[6:0], mosi};
      bitc++;
      if (bitc == 8) begin
        bitc = 0;
        got_byte(rx_sh);
        // keep a multi-block read going
        if (reading && out_q.size() < 4) begin
          push_block(raddr);
          raddr += 512;
        end
      end
    end
  end

  always @(negedge sclk) begin
    if (!cs_n && !power_n) begin
      if (bitc == 0) begin
        logic [8:0] item = (out_q.size() != 0) ? out_q.pop_front() : 9'h0FF;
        tx_sh = item[7:0];
        // after the last byte of a single-block read the card needs 8 more
        // clocks (this byte's 8 + 8) before the next command
        if (item[8] && !reading) settle = 16;
      end else begin
        tx_sh = {tx_sh[6:0], 1'b1};
      end
    end
  end

endmodule
