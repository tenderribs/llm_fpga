# Generate and synthesize the IP cores this design uses into build/ip/. Run through the
# Makefile (make ip); the synth stage reads the generated .xci files.
#
# mig_ddr3: 7-series DDR3 controller for the Genesys 2's 1 GB DDR3, set up
# from Digilent's board preset (src/ip/mig_ddr3.prj, a copy of
# board_support/vivado-boards/new/board_files/genesys2/H/mig.prj with the
# module name changed).

set repo_root [file normalize [file dirname [info script]]/..]
set ip_dir    $repo_root/build/ip
set part      xc7k325tffg900-2

file mkdir $ip_dir
create_project -in_memory -part $part
set_property target_language Verilog [current_project]

create_ip -name mig_7series -vendor xilinx.com -library ip -version 4.2 \
    -module_name mig_ddr3 -dir $ip_dir
set_property CONFIG.XML_INPUT_FILE $repo_root/src/ip/mig_ddr3.prj [get_ips mig_ddr3]
generate_target all [get_ips mig_ddr3]
# synthesize it out of context now, so design builds just link the checkpoint
synth_ip [get_ips mig_ddr3]
puts "generated [get_property IP_FILE [get_ips mig_ddr3]]"
