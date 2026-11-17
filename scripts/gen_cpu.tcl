# Generate the CPU subsystem block design (build/cpu_bd), used by top_soc.
# Run through the Makefile (make cpu).
#
#   MicroBlaze V: RV32IMF (single-precision FPU, multiply / divide),
#     frequency-optimized pipeline, 8 KB instruction + 8 KB data cache over
#     DDR, debug module (load and debug firmware over JTAG with xsdb)
#   128 KB local memory (LMB block RAM) for code, data and stack
#   SmartConnect: CPU caches, the CPU's uncached port and the external
#     master (llm_core's weight reader / SD / UART path) to DDR; the CPU's
#     uncached port also reaches llm_core's registers
#
# Address map (CPU and S_AXI_EXT view):
#   0x0000_0000  128 KB  local memory (CPU only)
#   0x4000_0000  256 KB  M_AXI_PERIPH: llm_core registers (CPU only)
#   0x8000_0000    1 GB  M_AXI_DDR: DDR3. The caches cover the lower 512 MB
#                        (0x8000_0000..0x9FFF_FFFF; the cached range must be
#                        a power of two); the upper 512 MB (0xA000_0000..)
#                        is reached uncached, through the peripheral port.
#
# External ports (all on clk, 225 MHz = the DDR controller's user clock):
#   clk, resetn, S_AXI_EXT (AXI4, 256-bit), M_AXI_DDR (AXI4, 256-bit),
#   M_AXI_PERIPH (AXI4-Lite, 32-bit)

set repo_root [file normalize [file dirname [info script]]/..]
set bd_dir    $repo_root/build/cpu_bd
set part      xc7k325tffg900-2
set clk_hz    225000000

file delete -force $bd_dir
create_project -force cpu_bd $bd_dir -part $part
set_property target_language Verilog [current_project]
create_bd_design cpu

# ---------------------------------------------------------------------------
# Ports
# ---------------------------------------------------------------------------
set clk [create_bd_port -dir I -type clk -freq_hz $clk_hz clk]
set rstn [create_bd_port -dir I -type rst resetn]
set_property CONFIG.POLARITY ACTIVE_LOW $rstn

set s_ext [create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 S_AXI_EXT]
set_property -dict [list CONFIG.PROTOCOL AXI4 CONFIG.DATA_WIDTH 256 CONFIG.ADDR_WIDTH 32 \
    CONFIG.ID_WIDTH 0 CONFIG.HAS_LOCK 0 CONFIG.HAS_CACHE 0 CONFIG.HAS_PROT 0 CONFIG.HAS_QOS 0 \
    CONFIG.HAS_REGION 0 CONFIG.HAS_WSTRB 1 CONFIG.HAS_BRESP 1 CONFIG.HAS_RRESP 1 \
    CONFIG.NUM_READ_OUTSTANDING 4 CONFIG.NUM_WRITE_OUTSTANDING 2 \
    CONFIG.FREQ_HZ $clk_hz] $s_ext

set m_ddr [create_bd_intf_port -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI_DDR]
set_property -dict [list CONFIG.PROTOCOL AXI4 CONFIG.DATA_WIDTH 256 CONFIG.ADDR_WIDTH 32 \
    CONFIG.FREQ_HZ $clk_hz] $m_ddr

set m_periph [create_bd_intf_port -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI_PERIPH]
set_property -dict [list CONFIG.PROTOCOL AXI4LITE CONFIG.DATA_WIDTH 32 CONFIG.ADDR_WIDTH 32 \
    CONFIG.FREQ_HZ $clk_hz] $m_periph

set_property CONFIG.ASSOCIATED_BUSIF {S_AXI_EXT:M_AXI_DDR:M_AXI_PERIPH} $clk
set_property CONFIG.ASSOCIATED_RESET {resetn} $clk

# ---------------------------------------------------------------------------
# CPU
# ---------------------------------------------------------------------------
set mbv [create_bd_cell -type ip -vlnv xilinx.com:ip:microblaze_riscv:1.0 mbv]
set_property -dict [list \
    CONFIG.C_USE_FPU 1 \
    CONFIG.C_USE_MULDIV 2 \
    CONFIG.C_OPTIMIZATION 2 \
    CONFIG.C_D_AXI 1 \
    CONFIG.C_D_LMB 1 \
    CONFIG.C_I_LMB 1 \
    CONFIG.C_USE_ICACHE 1 \
    CONFIG.C_USE_DCACHE 1 \
    CONFIG.C_ICACHE_BYTE_SIZE 8192 \
    CONFIG.C_DCACHE_BYTE_SIZE 8192 \
    CONFIG.C_ICACHE_BASEADDR 0x0000000080000000 \
    CONFIG.C_ICACHE_HIGHADDR 0x000000009FFFFFFF \
    CONFIG.C_DCACHE_BASEADDR 0x0000000080000000 \
    CONFIG.C_DCACHE_HIGHADDR 0x000000009FFFFFFF \
    CONFIG.C_DEBUG_ENABLED 1 \
] $mbv

set mdm [create_bd_cell -type ip -vlnv xilinx.com:ip:mdm_riscv:1.0 mdm]

set rst [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst]
set_property CONFIG.C_EXT_RST_WIDTH 1 $rst

# ---------------------------------------------------------------------------
# Local memory: one true dual-port block RAM, I side and D side
# ---------------------------------------------------------------------------
set ilmb [create_bd_cell -type ip -vlnv xilinx.com:ip:lmb_v10:3.0 ilmb]
set dlmb [create_bd_cell -type ip -vlnv xilinx.com:ip:lmb_v10:3.0 dlmb]
set ilmb_ctrl [create_bd_cell -type ip -vlnv xilinx.com:ip:lmb_bram_if_cntlr:4.0 ilmb_ctrl]
set dlmb_ctrl [create_bd_cell -type ip -vlnv xilinx.com:ip:lmb_bram_if_cntlr:4.0 dlmb_ctrl]
set lmem [create_bd_cell -type ip -vlnv xilinx.com:ip:blk_mem_gen:8.4 lmem]
set_property -dict [list CONFIG.Memory_Type True_Dual_Port_RAM CONFIG.use_bram_block BRAM_Controller] $lmem

connect_bd_intf_net [get_bd_intf_pins mbv/ILMB] [get_bd_intf_pins ilmb/LMB_M]
connect_bd_intf_net [get_bd_intf_pins mbv/DLMB] [get_bd_intf_pins dlmb/LMB_M]
connect_bd_intf_net [get_bd_intf_pins ilmb/LMB_Sl_0] [get_bd_intf_pins ilmb_ctrl/SLMB]
connect_bd_intf_net [get_bd_intf_pins dlmb/LMB_Sl_0] [get_bd_intf_pins dlmb_ctrl/SLMB]
connect_bd_intf_net [get_bd_intf_pins ilmb_ctrl/BRAM_PORT] [get_bd_intf_pins lmem/BRAM_PORTA]
connect_bd_intf_net [get_bd_intf_pins dlmb_ctrl/BRAM_PORT] [get_bd_intf_pins lmem/BRAM_PORTB]

# ---------------------------------------------------------------------------
# Interconnect
# ---------------------------------------------------------------------------
set sc [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 sc]
set_property -dict [list CONFIG.NUM_SI 4 CONFIG.NUM_MI 2] $sc

connect_bd_intf_net [get_bd_intf_pins mbv/M_AXI_IC] [get_bd_intf_pins sc/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins mbv/M_AXI_DC] [get_bd_intf_pins sc/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins mbv/M_AXI_DP] [get_bd_intf_pins sc/S02_AXI]
connect_bd_intf_net $s_ext [get_bd_intf_pins sc/S03_AXI]
connect_bd_intf_net [get_bd_intf_pins sc/M00_AXI] $m_ddr
connect_bd_intf_net [get_bd_intf_pins sc/M01_AXI] $m_periph

# ---------------------------------------------------------------------------
# Clocks, resets, debug
# ---------------------------------------------------------------------------
connect_bd_net $clk [get_bd_pins mbv/Clk] [get_bd_pins ilmb/LMB_Clk] [get_bd_pins dlmb/LMB_Clk] \
    [get_bd_pins ilmb_ctrl/LMB_Clk] [get_bd_pins dlmb_ctrl/LMB_Clk] [get_bd_pins sc/aclk] \
    [get_bd_pins rst/slowest_sync_clk]
connect_bd_net $rstn [get_bd_pins rst/ext_reset_in]
connect_bd_net [get_bd_pins mdm/Debug_SYS_Rst] [get_bd_pins rst/mb_debug_sys_rst]
connect_bd_net [get_bd_pins rst/mb_reset] [get_bd_pins mbv/Reset]
connect_bd_net [get_bd_pins rst/bus_struct_reset] [get_bd_pins ilmb/SYS_Rst] [get_bd_pins dlmb/SYS_Rst] \
    [get_bd_pins ilmb_ctrl/LMB_Rst] [get_bd_pins dlmb_ctrl/LMB_Rst]
connect_bd_net [get_bd_pins rst/interconnect_aresetn] [get_bd_pins sc/aresetn]
connect_bd_intf_net [get_bd_intf_pins mdm/MBDEBUG_0] [get_bd_intf_pins mbv/DEBUG]

# ---------------------------------------------------------------------------
# Addresses
# ---------------------------------------------------------------------------
assign_bd_address -offset 0x00000000 -range 128K [get_bd_addr_segs ilmb_ctrl/SLMB/Mem]
assign_bd_address -offset 0x00000000 -range 128K [get_bd_addr_segs dlmb_ctrl/SLMB/Mem]
foreach space {mbv/Instruction mbv/Data} {
    assign_bd_address -target_address_space [get_bd_addr_spaces $space] \
        -offset 0x80000000 -range 1G [get_bd_addr_segs M_AXI_DDR/Reg]
}
assign_bd_address -target_address_space [get_bd_addr_spaces mbv/Data] \
    -offset 0x40000000 -range 256K [get_bd_addr_segs M_AXI_PERIPH/Reg]
assign_bd_address -target_address_space [get_bd_addr_spaces S_AXI_EXT] \
    -offset 0x80000000 -range 1G [get_bd_addr_segs M_AXI_DDR/Reg]

# the external master and instruction fetches don't reach the registers
exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces S_AXI_EXT] \
    [get_bd_addr_segs M_AXI_PERIPH/Reg]
exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces mbv/Instruction] \
    [get_bd_addr_segs M_AXI_PERIPH/Reg]

validate_bd_design
save_bd_design

# synthesized together with the rest of the design (no separate checkpoints)
set bd_file [get_files cpu.bd]
set_property synth_checkpoint_mode None $bd_file
generate_target all $bd_file
puts "generated $bd_file"
