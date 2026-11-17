`default_nettype none

// GEMM over UART: Y = A @ B + C on an N x N systolic array.
//
// Protocol (8N1, Baud):
//   host -> FPGA: A, B, C, each N*N signed bytes, row-major (3*N*N bytes)
//   FPGA -> host: Y, N*N signed 32-bit little-endian words, row-major
//                 (4*N*N bytes)
// If the host stops partway through sending a job for IdleTimeoutClks, the
// partial job is discarded and the next byte starts a new job. Bytes that
// arrive while a job is computing or sending are dropped.
//
// LEDs: 0 heartbeat, 1 receiving, 2 computing, 3 sending,
//       7:4 count of completed jobs (mod 16)
module top_gemm #(
    parameter int N = 16,
    parameter int ClkHz = 200_000_000,
    parameter int Baud = 115_200,
    parameter int IdleTimeoutClks = ClkHz / 10
) (
    input wire logic sysclk_p,
    input wire logic sysclk_n,
    input wire logic cpu_resetn,
    input wire logic uart_rx_i,
    output logic uart_tx_o,
    output logic [7:0] led_o
);

  localparam int ClksPerBit = ClkHz / Baud;
  localparam int MatBytes = N * N;
  localparam int InBytes = 3 * MatBytes;
  localparam int OutBytes = 4 * MatBytes;
  localparam int CntW = $clog2(OutBytes);

  // ---------------------------------------------------------------------------
  // Clock and reset
  // ---------------------------------------------------------------------------
  logic clk_ibuf, clk;

  IBUFDS u_clk_ibuf (
      .I (sysclk_p),
      .IB(sysclk_n),
      .O (clk_ibuf)
  );

  BUFG u_clk_bufg (
      .I(clk_ibuf),
      .O(clk)
  );

  // Button is asynchronous: assert reset immediately, release it synchronously
  logic [1:0] rst_sync_q;
  logic rst_n;
  assign rst_n = rst_sync_q[1];

  always_ff @(posedge clk, negedge cpu_resetn) begin
    if (~cpu_resetn) rst_sync_q <= '0;
    else rst_sync_q <= {rst_sync_q[0], 1'b1};
  end

  // ---------------------------------------------------------------------------
  // UART
  // ---------------------------------------------------------------------------
  logic [7:0] rx_data;
  logic rx_valid;
  logic tx_valid, tx_ready;

  uart_rx #(
      .ClksPerBit(ClksPerBit)
  ) u_uart_rx (
      .clk_i  (clk),
      .rst_ni (rst_n),
      .rx_i   (uart_rx_i),
      .data_o (rx_data),
      .valid_o(rx_valid)
  );

  // ---------------------------------------------------------------------------
  // Matrix buffers
  // ---------------------------------------------------------------------------
  // Input bytes shift in from the top, so after InBytes bytes the first byte
  // received sits at index 0: in_q = {C, B, A}, each row-major.
  logic [InBytes-1:0][7:0] in_q;
  // Y rows shift in from the top the same way, then shift out one byte at a
  // time from index 0. The packed layout puts y[0][0] bits 7:0 at byte 0, so
  // the bytes leave in row-major little-endian order.
  logic [OutBytes-1:0][7:0] y_q;

  logic signed [7:0] a[N][N];
  logic signed [7:0] b[N][N];
  logic signed [7:0] c[N][N];
  logic signed [31:0] y_row[N];
  logic [N-1:0][31:0] y_row_packed;

  for (genvar i = 0; i < N; i++) begin : g_unpack_row
    for (genvar j = 0; j < N; j++) begin : g_unpack_col
      assign a[i][j] = in_q[0*MatBytes+i*N+j];
      assign b[i][j] = in_q[1*MatBytes+i*N+j];
      assign c[i][j] = in_q[2*MatBytes+i*N+j];
    end
    assign y_row_packed[i] = y_row[i];
  end

  // ---------------------------------------------------------------------------
  // Systolic array
  // ---------------------------------------------------------------------------
  logic sa_valid, sa_ready, sa_y_valid;

  systolic_array #(
      .N(N)
  ) u_sa (
      .clk_i    (clk),
      .rst_ni   (rst_n),
      .valid_i  (sa_valid),
      .a_i      (a),
      .b_i      (b),
      .c_i      (c),
      .y_o      (y_row),
      .ready_o  (sa_ready),
      .y_valid_o(sa_y_valid)
  );

  // ---------------------------------------------------------------------------
  // Control
  // ---------------------------------------------------------------------------
  typedef enum logic [1:0] {
    RECV,
    START,
    COMPUTE,
    SEND
  } state_t;
  state_t state_d, state_q;

  // Counts bytes received (RECV), rows captured (COMPUTE) or bytes sent (SEND)
  logic [CntW-1:0] cnt_d, cnt_q;
  logic [$clog2(IdleTimeoutClks)-1:0] idle_d, idle_q;
  logic [3:0] jobs_d, jobs_q;

  logic shift_in, capture_row, shift_out;

  always_comb begin
    state_d = state_q;
    cnt_d = cnt_q;
    idle_d = idle_q;
    jobs_d = jobs_q;

    shift_in = 1'b0;
    capture_row = 1'b0;
    shift_out = 1'b0;
    sa_valid = 1'b0;
    tx_valid = 1'b0;

    case (state_q)
      RECV: begin
        if (rx_valid) begin
          shift_in = 1'b1;
          idle_d = '0;
          if (cnt_q == CntW'(InBytes - 1)) begin
            cnt_d = '0;
            state_d = START;
          end else begin
            cnt_d = cnt_q + 1;
          end
        end else if (cnt_q != '0) begin
          // Partway through a job: drop it if the host has gone quiet
          if (idle_q == ($clog2(IdleTimeoutClks))'(IdleTimeoutClks - 1)) begin
            cnt_d  = '0;
            idle_d = '0;
          end else begin
            idle_d = idle_q + 1;
          end
        end
      end

      START: begin
        sa_valid = 1'b1;
        if (sa_ready) state_d = COMPUTE;
      end

      COMPUTE: begin
        if (sa_y_valid) begin
          capture_row = 1'b1;
          if (cnt_q == CntW'(N - 1)) begin
            cnt_d = '0;
            state_d = SEND;
          end else begin
            cnt_d = cnt_q + 1;
          end
        end
      end

      SEND: begin
        tx_valid = 1'b1;
        if (tx_ready) begin
          shift_out = 1'b1;
          if (cnt_q == CntW'(OutBytes - 1)) begin
            cnt_d = '0;
            jobs_d = jobs_q + 1;
            state_d = RECV;
          end else begin
            cnt_d = cnt_q + 1;
          end
        end
      end

      default: state_d = RECV;
    endcase
  end

  always_ff @(posedge clk, negedge rst_n) begin
    if (~rst_n) begin
      state_q <= RECV;
      cnt_q <= '0;
      idle_q <= '0;
      jobs_q <= '0;
    end else begin
      state_q <= state_d;
      cnt_q <= cnt_d;
      idle_q <= idle_d;
      jobs_q <= jobs_d;
    end
  end

  // Data registers: no reset, so they can be packed densely
  always_ff @(posedge clk) begin
    if (shift_in) in_q <= {rx_data, in_q[InBytes-1:1]};

    if (capture_row) y_q <= {y_row_packed, y_q[OutBytes-1:N*4]};
    else if (shift_out) y_q <= {8'h00, y_q[OutBytes-1:1]};
  end

  uart_tx #(
      .ClksPerBit(ClksPerBit)
  ) u_uart_tx (
      .clk_i  (clk),
      .rst_ni (rst_n),
      .data_i (y_q[0]),
      .valid_i(tx_valid),
      .ready_o(tx_ready),
      .tx_o   (uart_tx_o)
  );

  // ---------------------------------------------------------------------------
  // LEDs
  // ---------------------------------------------------------------------------
  logic [26:0] heartbeat_q;

  always_ff @(posedge clk, negedge rst_n) begin
    if (~rst_n) heartbeat_q <= '0;
    else heartbeat_q <= heartbeat_q + 1;
  end

  always_ff @(posedge clk) begin
    led_o[0]   <= heartbeat_q[26];
    led_o[1]   <= (state_q == RECV) && (cnt_q != '0);
    led_o[2]   <= (state_q == START) || (state_q == COMPUTE);
    led_o[3]   <= (state_q == SEND);
    led_o[7:4] <= jobs_q;
  end

endmodule

`default_nettype wire
