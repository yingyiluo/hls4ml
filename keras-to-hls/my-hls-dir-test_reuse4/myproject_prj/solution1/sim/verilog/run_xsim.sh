
/data/xilinx/Vivado/2016.4/bin/xelab xil_defaultlib.apatb_myproject_top -prj myproject.prj --initfile "/data/xilinx/Vivado/2016.4/data/xsim/ip/xsim_ip.ini" --lib "ieee_proposed=./ieee_proposed" -s myproject -debug wave
/data/xilinx/Vivado/2016.4/bin/xsim --noieeewarnings myproject -tclbatch myproject.tcl

