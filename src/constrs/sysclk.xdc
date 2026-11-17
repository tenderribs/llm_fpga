## 200 MHz board clock, for designs that use it directly (top_gemm, top_qmv).
## top_llm doesn't read this file: its DDR controller takes the clock and
## constrains these pins itself.
## 200 MHz system clock
set_property -dict { PACKAGE_PIN AD11  IOSTANDARD LVDS     } [get_ports { sysclk_n }]; #IO_L12N_T1_MRCC_33 Sch=sysclk_n
set_property -dict { PACKAGE_PIN AD12  IOSTANDARD LVDS     } [get_ports { sysclk_p }]; #IO_L12P_T1_MRCC_33 Sch=sysclk_p
create_clock -period 5.000 -name sysclk [get_ports sysclk_p]

