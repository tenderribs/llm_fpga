// Simulation stand-in for the Xilinx BUFG primitive, for simulators without
// the Vivado unisims library (e.g. Verilator). Passes the input through.
module BUFG (
    input  wire I,
    output wire O
);
  assign O = I;
endmodule
