`default_nettype none

// 8N1 UART receiver. Samples each bit once, in the middle of the bit period.
// valid_o pulses for one cycle with data_o when a byte with a good stop bit
// has been received; bytes with a bad stop bit are dropped.
module uart_rx #(
    parameter int ClksPerBit
) (
    input wire logic clk_i,
    input wire logic rst_ni,
    input wire logic rx_i,
    output logic [7:0] data_o,
    output logic valid_o
);

  // rx_i is asynchronous to clk_i: two flops before anything looks at it
  (* ASYNC_REG = "TRUE" *) logic [1:0] rx_sync_q;
  logic rx;
  assign rx = rx_sync_q[1];

  typedef enum logic [1:0] {
    IDLE,
    START,
    DATA,
    STOP
  } state_t;
  state_t state_q;

  logic [$clog2(ClksPerBit)-1:0] clk_cnt_q;
  logic [2:0] bit_idx_q;
  logic [7:0] data_q;

  assign data_o = data_q;

  always_ff @(posedge clk_i, negedge rst_ni) begin
    if (~rst_ni) begin
      rx_sync_q <= '1;
      state_q <= IDLE;
      clk_cnt_q <= '0;
      bit_idx_q <= '0;
      data_q <= '0;
      valid_o <= 1'b0;
    end else begin
      rx_sync_q <= {rx_sync_q[0], rx_i};
      valid_o <= 1'b0;

      case (state_q)
        IDLE: begin
          clk_cnt_q <= '0;
          if (~rx) state_q <= START;
        end

        // Wait half a bit, then check the line is still low (not a glitch)
        START: begin
          if (clk_cnt_q == ($clog2(ClksPerBit))'(ClksPerBit / 2 - 1)) begin
            clk_cnt_q <= '0;
            bit_idx_q <= '0;
            state_q <= rx ? IDLE : DATA;
          end else begin
            clk_cnt_q <= clk_cnt_q + 1;
          end
        end

        // From the middle of the start bit, every full bit period lands in
        // the middle of the next bit. LSB first.
        DATA: begin
          if (clk_cnt_q == ($clog2(ClksPerBit))'(ClksPerBit - 1)) begin
            clk_cnt_q <= '0;
            data_q <= {rx, data_q[7:1]};
            bit_idx_q <= bit_idx_q + 1;
            if (bit_idx_q == 3'd7) state_q <= STOP;
          end else begin
            clk_cnt_q <= clk_cnt_q + 1;
          end
        end

        // Back to IDLE in the middle of the stop bit; IDLE waits for the
        // next falling edge.
        STOP: begin
          if (clk_cnt_q == ($clog2(ClksPerBit))'(ClksPerBit - 1)) begin
            valid_o <= rx;
            state_q <= IDLE;
          end else begin
            clk_cnt_q <= clk_cnt_q + 1;
          end
        end

        default: state_q <= IDLE;
      endcase
    end
  end

endmodule

`default_nettype wire
