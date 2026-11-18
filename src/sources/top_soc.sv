`default_nettype none

// The full system: MicroBlaze V CPU (block design `cpu`, scripts/gen_cpu.tcl)
// + llm_core (matmul engine, microSD, UART commands) + DDR3 controller.
//
//   llm_core's DDR master ---> cpu/S_AXI_EXT \
//   CPU caches, uncached port ------------------> SmartConnect --> DDR3
//   CPU uncached port -------------------------> llm_core registers
//
// llm_core and the DDR side run on the DDR controller's 225 MHz user clock;
// the CPU runs at 150 MHz from a clocking wizard in the block design, and
// the SmartConnect and the dual-port Y / V RAMs cross between the two. The
// UART is shared: host commands as in top_llm, and the firmware's console
// output in between (see llm_core.sv). The firmware is built into the
// bitstream (make TOP=top_soc fwbit) or loaded over JTAG with xsdb.
module top_soc #(
    parameter int Baud = 3_000_000
) (
    input wire logic sysclk_p,
    input wire logic sysclk_n,
    input wire logic cpu_resetn,
    input wire logic uart_rx_i,
    output logic uart_tx_o,
    output logic [7:0] led_o,

    inout wire [31:0] ddr3_dq,
    inout wire [3:0] ddr3_dqs_n,
    inout wire [3:0] ddr3_dqs_p,
    output logic [14:0] ddr3_addr,
    output logic [2:0] ddr3_ba,
    output logic ddr3_ras_n,
    output logic ddr3_cas_n,
    output logic ddr3_we_n,
    output logic ddr3_reset_n,
    output logic [0:0] ddr3_ck_p,
    output logic [0:0] ddr3_ck_n,
    output logic [0:0] ddr3_cke,
    output logic [0:0] ddr3_cs_n,
    output logic [3:0] ddr3_dm,
    output logic [0:0] ddr3_odt,

    output logic sd_reset,
    output logic sd_sclk,
    output logic sd_cmd,
    output logic sd_dat3,
    input wire logic sd_dat0,
    input wire logic sd_cd
);

  localparam int AddrW = 30;

  logic ui_clk, ui_rst, calib_done, mmcm_locked;
  logic aresetn_q, rst_n_q;

  always_ff @(posedge ui_clk) begin
    aresetn_q <= ~ui_rst;
    rst_n_q   <= ~ui_rst;
  end

  // ---------------------------------------------------------------------------
  // llm_core <-> CPU block design
  // ---------------------------------------------------------------------------
  // llm_core's DDR master
  logic [AddrW-1:0] c_awaddr, c_araddr;
  logic [7:0] c_awlen, c_arlen;
  logic c_awvalid, c_awready, c_wlast, c_wvalid, c_wready, c_bvalid, c_bready;
  logic c_arvalid, c_arready, c_rvalid, c_rready;
  logic [255:0] c_wdata, c_rdata;
  logic [31:0] c_wstrb;

  // CPU local memory bus -> llm_core's Y RAM (read) and V RAM (read / write)
  logic yb_clk, yb_en;
  logic [3:0] yb_we;
  logic [31:0] yb_addr, yb_din, yb_dout;

  // CPU -> llm_core registers
  logic [31:0] p_awaddr, p_araddr, p_wdata, p_rdata;
  logic p_awvalid, p_awready, p_wvalid, p_wready, p_bvalid, p_bready;
  logic p_arvalid, p_arready, p_rvalid, p_rready;

  // SmartConnect -> DDR controller
  logic [31:0] d_awaddr, d_araddr;
  logic [7:0] d_awlen, d_arlen;
  logic [2:0] d_awsize, d_arsize, d_awprot, d_arprot;
  logic [1:0] d_awburst, d_arburst, d_bresp, d_rresp;
  logic [3:0] d_awcache, d_arcache, d_awqos, d_arqos;
  logic [0:0] d_awlock, d_arlock;
  logic d_awvalid, d_awready, d_wlast, d_wvalid, d_wready, d_bvalid, d_bready;
  logic d_arvalid, d_arready, d_rlast, d_rvalid, d_rready;
  logic [255:0] d_wdata, d_rdata;
  logic [31:0] d_wstrb;

  cpu u_cpu (
      .clk(ui_clk),
      .resetn(rst_n_q),

      // llm_core's DDR traffic: DDR is at 0x8000_0000 in the CPU's map
      .S_AXI_EXT_awaddr({2'b10, c_awaddr}),
      .S_AXI_EXT_awlen(c_awlen),
      .S_AXI_EXT_awsize(3'd5),
      .S_AXI_EXT_awburst(2'b01),
      .S_AXI_EXT_awlock(1'b0),
      .S_AXI_EXT_awcache(4'b0011),
      .S_AXI_EXT_awprot(3'b000),
      .S_AXI_EXT_awqos(4'd0),
      .S_AXI_EXT_awvalid(c_awvalid),
      .S_AXI_EXT_awready(c_awready),
      .S_AXI_EXT_wdata(c_wdata),
      .S_AXI_EXT_wstrb(c_wstrb),
      .S_AXI_EXT_wlast(c_wlast),
      .S_AXI_EXT_wvalid(c_wvalid),
      .S_AXI_EXT_wready(c_wready),
      .S_AXI_EXT_bresp(),
      .S_AXI_EXT_bvalid(c_bvalid),
      .S_AXI_EXT_bready(c_bready),
      .S_AXI_EXT_araddr({2'b10, c_araddr}),
      .S_AXI_EXT_arlen(c_arlen),
      .S_AXI_EXT_arsize(3'd5),
      .S_AXI_EXT_arburst(2'b01),
      .S_AXI_EXT_arlock(1'b0),
      .S_AXI_EXT_arcache(4'b0011),
      .S_AXI_EXT_arprot(3'b000),
      .S_AXI_EXT_arqos(4'd0),
      .S_AXI_EXT_arvalid(c_arvalid),
      .S_AXI_EXT_arready(c_arready),
      .S_AXI_EXT_rdata(c_rdata),
      .S_AXI_EXT_rresp(),
      .S_AXI_EXT_rlast(),
      .S_AXI_EXT_rvalid(c_rvalid),
      .S_AXI_EXT_rready(c_rready),

      .M_AXI_PERIPH_awaddr(p_awaddr),
      .M_AXI_PERIPH_awprot(),
      .M_AXI_PERIPH_awvalid(p_awvalid),
      .M_AXI_PERIPH_awready(p_awready),
      .M_AXI_PERIPH_wdata(p_wdata),
      .M_AXI_PERIPH_wstrb(),
      .M_AXI_PERIPH_wvalid(p_wvalid),
      .M_AXI_PERIPH_wready(p_wready),
      .M_AXI_PERIPH_bresp(2'b00),
      .M_AXI_PERIPH_bvalid(p_bvalid),
      .M_AXI_PERIPH_bready(p_bready),
      .M_AXI_PERIPH_araddr(p_araddr),
      .M_AXI_PERIPH_arprot(),
      .M_AXI_PERIPH_arvalid(p_arvalid),
      .M_AXI_PERIPH_arready(p_arready),
      .M_AXI_PERIPH_rdata(p_rdata),
      .M_AXI_PERIPH_rresp(2'b00),
      .M_AXI_PERIPH_rvalid(p_rvalid),
      .M_AXI_PERIPH_rready(p_rready),

      // llm_core's Y RAM (read) and V RAM (read / write) on the CPU's local memory bus
      .Y_BRAM_addr(yb_addr),
      .Y_BRAM_clk(yb_clk),
      .Y_BRAM_din(yb_din),
      .Y_BRAM_dout(yb_dout),
      .Y_BRAM_en(yb_en),
      .Y_BRAM_rst(),
      .Y_BRAM_we(yb_we),

      .M_AXI_DDR_awaddr(d_awaddr),
      .M_AXI_DDR_awlen(d_awlen),
      .M_AXI_DDR_awsize(d_awsize),
      .M_AXI_DDR_awburst(d_awburst),
      .M_AXI_DDR_awlock(d_awlock),
      .M_AXI_DDR_awcache(d_awcache),
      .M_AXI_DDR_awprot(d_awprot),
      .M_AXI_DDR_awqos(d_awqos),
      .M_AXI_DDR_awvalid(d_awvalid),
      .M_AXI_DDR_awready(d_awready),
      .M_AXI_DDR_wdata(d_wdata),
      .M_AXI_DDR_wstrb(d_wstrb),
      .M_AXI_DDR_wlast(d_wlast),
      .M_AXI_DDR_wvalid(d_wvalid),
      .M_AXI_DDR_wready(d_wready),
      .M_AXI_DDR_bresp(d_bresp),
      .M_AXI_DDR_bvalid(d_bvalid),
      .M_AXI_DDR_bready(d_bready),
      .M_AXI_DDR_araddr(d_araddr),
      .M_AXI_DDR_arlen(d_arlen),
      .M_AXI_DDR_arsize(d_arsize),
      .M_AXI_DDR_arburst(d_arburst),
      .M_AXI_DDR_arlock(d_arlock),
      .M_AXI_DDR_arcache(d_arcache),
      .M_AXI_DDR_arprot(d_arprot),
      .M_AXI_DDR_arqos(d_arqos),
      .M_AXI_DDR_arvalid(d_arvalid),
      .M_AXI_DDR_arready(d_arready),
      .M_AXI_DDR_rdata(d_rdata),
      .M_AXI_DDR_rresp(d_rresp),
      .M_AXI_DDR_rlast(d_rlast),
      .M_AXI_DDR_rvalid(d_rvalid),
      .M_AXI_DDR_rready(d_rready)
  );

  llm_core #(
      .ClkHz(225_000_000),
      .Baud (Baud),
      .AddrW(AddrW)
  ) u_core (
      .clk_i(ui_clk),
      .rst_ni(rst_n_q),
      .calib_done_i(calib_done),
      .uart_rx_i(uart_rx_i),
      .uart_tx_o(uart_tx_o),
      .led_o(led_o),
      .m_awaddr_o(c_awaddr),
      .m_awlen_o(c_awlen),
      .m_awvalid_o(c_awvalid),
      .m_awready_i(c_awready),
      .m_wdata_o(c_wdata),
      .m_wstrb_o(c_wstrb),
      .m_wlast_o(c_wlast),
      .m_wvalid_o(c_wvalid),
      .m_wready_i(c_wready),
      .m_bvalid_i(c_bvalid),
      .m_bready_o(c_bready),
      .m_araddr_o(c_araddr),
      .m_arlen_o(c_arlen),
      .m_arvalid_o(c_arvalid),
      .m_arready_i(c_arready),
      .m_rdata_i(c_rdata),
      .m_rvalid_i(c_rvalid),
      .m_rready_o(c_rready),
      .sd_power_n_o(sd_reset),
      .sd_sclk_o(sd_sclk),
      .sd_cs_o(sd_dat3),
      .sd_mosi_o(sd_cmd),
      .sd_miso_i(sd_dat0),
      .sd_cd_i(sd_cd),
      .s_awaddr_i(p_awaddr[17:0]),
      .s_awvalid_i(p_awvalid),
      .s_awready_o(p_awready),
      .s_wdata_i(p_wdata),
      .s_wvalid_i(p_wvalid),
      .s_wready_o(p_wready),
      .s_bvalid_o(p_bvalid),
      .s_bready_i(p_bready),
      .s_araddr_i(p_araddr[17:0]),
      .s_arvalid_i(p_arvalid),
      .s_arready_o(p_arready),
      .s_rdata_o(p_rdata),
      .s_rvalid_o(p_rvalid),
      .s_rready_i(p_rready),
      .yb_clk_i(yb_clk),
      .yb_en_i(yb_en),
      .yb_we_i(yb_we),
      .yb_addr_i(yb_addr),
      .yb_din_i(yb_din),
      .yb_dout_o(yb_dout)
  );

  // ---------------------------------------------------------------------------
  // DDR3 controller
  // ---------------------------------------------------------------------------
  mig_ddr3 u_mig (
      .ddr3_dq(ddr3_dq),
      .ddr3_dqs_n(ddr3_dqs_n),
      .ddr3_dqs_p(ddr3_dqs_p),
      .ddr3_addr(ddr3_addr),
      .ddr3_ba(ddr3_ba),
      .ddr3_ras_n(ddr3_ras_n),
      .ddr3_cas_n(ddr3_cas_n),
      .ddr3_we_n(ddr3_we_n),
      .ddr3_reset_n(ddr3_reset_n),
      .ddr3_ck_p(ddr3_ck_p),
      .ddr3_ck_n(ddr3_ck_n),
      .ddr3_cke(ddr3_cke),
      .ddr3_cs_n(ddr3_cs_n),
      .ddr3_dm(ddr3_dm),
      .ddr3_odt(ddr3_odt),
      .sys_clk_p(sysclk_p),
      .sys_clk_n(sysclk_n),
      .ui_clk(ui_clk),
      .ui_clk_sync_rst(ui_rst),
      .mmcm_locked(mmcm_locked),
      .aresetn(aresetn_q),
      .app_sr_req(1'b0),
      .app_ref_req(1'b0),
      .app_zq_req(1'b0),
      .app_sr_active(),
      .app_ref_ack(),
      .app_zq_ack(),
      .s_axi_awid(3'd0),
      .s_axi_awaddr(d_awaddr[AddrW-1:0]),
      .s_axi_awlen(d_awlen),
      .s_axi_awsize(d_awsize),
      .s_axi_awburst(d_awburst),
      .s_axi_awlock(d_awlock),
      .s_axi_awcache(d_awcache),
      .s_axi_awprot(d_awprot),
      .s_axi_awqos(d_awqos),
      .s_axi_awvalid(d_awvalid),
      .s_axi_awready(d_awready),
      .s_axi_wdata(d_wdata),
      .s_axi_wstrb(d_wstrb),
      .s_axi_wlast(d_wlast),
      .s_axi_wvalid(d_wvalid),
      .s_axi_wready(d_wready),
      .s_axi_bready(d_bready),
      .s_axi_bid(),
      .s_axi_bresp(d_bresp),
      .s_axi_bvalid(d_bvalid),
      .s_axi_arid(3'd0),
      .s_axi_araddr(d_araddr[AddrW-1:0]),
      .s_axi_arlen(d_arlen),
      .s_axi_arsize(d_arsize),
      .s_axi_arburst(d_arburst),
      .s_axi_arlock(d_arlock),
      .s_axi_arcache(d_arcache),
      .s_axi_arprot(d_arprot),
      .s_axi_arqos(d_arqos),
      .s_axi_arvalid(d_arvalid),
      .s_axi_arready(d_arready),
      .s_axi_rready(d_rready),
      .s_axi_rid(),
      .s_axi_rdata(d_rdata),
      .s_axi_rresp(d_rresp),
      .s_axi_rlast(d_rlast),
      .s_axi_rvalid(d_rvalid),
      .init_calib_complete(calib_done),
      .device_temp(),
      .sys_rst(cpu_resetn)
  );

endmodule

`default_nettype wire
