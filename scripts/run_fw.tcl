# Load firmware into top_soc's MicroBlaze V over JTAG and start it.
#   xsdb scripts/run_fw.tcl <program.elf>
# The board must already be programmed with top_soc (make TOP=top_soc
# program). The program's console output comes out on the UART
# (host/console.py).

set elf [lindex $argv 0]
if {$elf eq ""} { error "usage: xsdb scripts/run_fw.tcl <program.elf>" }

connect
# The JTAG chip shows up as two targets (see make program); the debug module
# is on the one with the FPGA. Look for the CPU in all of them.
set cpu ""
foreach line [split [targets] "\n"] {
    if {[regexp {^\s*\*?\s*(\d+)\s+(.*)$} $line -> id name]} {
        if {[string match -nocase "*MicroBlaze*" $name] || [string match -nocase "*Hart*" $name]} {
            set cpu $id
            puts "CPU target $id: [string trim $name]"
            break
        }
    }
}
if {$cpu eq ""} {
    puts [targets]
    error "no MicroBlaze V target found"
}
targets $cpu
catch {stop}
rst -processor
dow $elf
con
puts "started $elf"
after 500
disconnect
