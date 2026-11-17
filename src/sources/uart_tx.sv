`default_nettype none

// 8N1 UART transmitter with a valid/ready handshake: a byte is taken when
// valid_i && ready_o.
module uart_tx #(
    parameter int ClksPerBit
) (
    input wire logic clk_i,
    input wire logic rst_ni,
    input wire logic [7:0] data_i,
    input wire logic valid_i,
    output logic ready_o,
    output logic tx_o
);

  // Frame shifts out LSB first: start bit (0), 8 data bits, stop bit (1)
  logic [9:0] frame_q;
  logic [3:0] bits_left_q;
  logic [$clog2(ClksPerBit)-1:0] clk_cnt_q;

  assign ready_o = (bits_left_q == '0);

  always_ff @(posedge clk_i, negedge rst_ni) begin
    if (~rst_ni) begin
      frame_q <= '1;
      bits_left_q <= '0;
      clk_cnt_q <= '0;
      tx_o <= 1'b1;
    end else begin
      if (ready_o) begin
        tx_o <= 1'b1;
        if (valid_i) begin
          frame_q <= {1'b1, data_i, 1'b0};
          bits_left_q <= 4'd10;
          clk_cnt_q <= '0;
        end
      end else begin
        tx_o <= frame_q[0];
        if (clk_cnt_q == ($clog2(ClksPerBit))'(ClksPerBit - 1)) begin
          clk_cnt_q <= '0;
          frame_q <= {1'b1, frame_q[9:1]};
          bits_left_q <= bits_left_q - 1;
        end else begin
          clk_cnt_q <= clk_cnt_q + 1;
        end
      end
    end
  end

endmodule

`default_nettype wire
