## top_llm only (scripts/build.tcl reads this file for that design).
## Pins from board_support/digilent-xdc/Genesys-2-Master.xdc.

## microSD slot, SPI mode: DAT3 = chip select, CMD = MOSI, DAT0 = MISO
set_property -dict { PACKAGE_PIN P28   IOSTANDARD LVCMOS33 } [get_ports { sd_cd }];   #IO_L8N_T1_D12_14 Sch=sd_cd
set_property -dict { PACKAGE_PIN R29   IOSTANDARD LVCMOS33 } [get_ports { sd_cmd }];  #IO_L7N_T1_D10_14 Sch=sd_cmd
set_property -dict { PACKAGE_PIN R26   IOSTANDARD LVCMOS33 PULLUP TRUE } [get_ports { sd_dat0 }]; #IO_L10N_T1_D15_14 Sch=sd_dat[0]
set_property -dict { PACKAGE_PIN T30   IOSTANDARD LVCMOS33 } [get_ports { sd_dat3 }]; #IO_L9N_T1_DQS_D13_14 Sch=sd_dat[3]
set_property -dict { PACKAGE_PIN AE24  IOSTANDARD LVCMOS33 } [get_ports { sd_reset }]; #IO_L12N_T1_MRCC_12 Sch=sd_reset
set_property -dict { PACKAGE_PIN R28   IOSTANDARD LVCMOS33 } [get_ports { sd_sclk }]; #IO_L11P_T1_SRCC_14 Sch=sd_sclk

## The SD clock is at most 22.5 MHz and generated from registers; MISO goes
## through a synchronizer. No I/O timing to check.
set_false_path -from [get_ports { sd_dat0 sd_cd }]
set_false_path -to   [get_ports { sd_cmd sd_dat3 sd_reset sd_sclk }]
