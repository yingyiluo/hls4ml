
log_wave -r /
set designtopgroup [add_wave_group "Design Top Signals"]
set coutputgroup [add_wave_group "C Outputs" -into $designtopgroup]
set const_size_out_group [add_wave_group const_size_out(wire) -into $coutputgroup]
add_wave /apatb_myproject_top/AESL_inst_myproject/const_size_out_ap_vld -into $const_size_out_group -color #ffff00 -radix hex
add_wave /apatb_myproject_top/AESL_inst_myproject/const_size_out -into $const_size_out_group -radix hex
set const_size_in_group [add_wave_group const_size_in(wire) -into $coutputgroup]
add_wave /apatb_myproject_top/AESL_inst_myproject/const_size_in_ap_vld -into $const_size_in_group -color #ffff00 -radix hex
add_wave /apatb_myproject_top/AESL_inst_myproject/const_size_in -into $const_size_in_group -radix hex
set res_group [add_wave_group res(wire) -into $coutputgroup]
add_wave /apatb_myproject_top/AESL_inst_myproject/res_0_V_ap_vld -into $res_group -color #ffff00 -radix hex
add_wave /apatb_myproject_top/AESL_inst_myproject/res_0_V -into $res_group -radix hex
set cinputgroup [add_wave_group "C Inputs" -into $designtopgroup]
set data_group [add_wave_group data(wire) -into $cinputgroup]
add_wave /apatb_myproject_top/AESL_inst_myproject/data_9_V -into $data_group -radix hex
add_wave /apatb_myproject_top/AESL_inst_myproject/data_8_V -into $data_group -radix hex
add_wave /apatb_myproject_top/AESL_inst_myproject/data_7_V -into $data_group -radix hex
add_wave /apatb_myproject_top/AESL_inst_myproject/data_6_V -into $data_group -radix hex
add_wave /apatb_myproject_top/AESL_inst_myproject/data_5_V -into $data_group -radix hex
add_wave /apatb_myproject_top/AESL_inst_myproject/data_4_V -into $data_group -radix hex
add_wave /apatb_myproject_top/AESL_inst_myproject/data_3_V -into $data_group -radix hex
add_wave /apatb_myproject_top/AESL_inst_myproject/data_2_V -into $data_group -radix hex
add_wave /apatb_myproject_top/AESL_inst_myproject/data_1_V -into $data_group -radix hex
add_wave /apatb_myproject_top/AESL_inst_myproject/data_0_V -into $data_group -radix hex
set blocksiggroup [add_wave_group "Block-level IO Handshake" -into $designtopgroup]
add_wave /apatb_myproject_top/AESL_inst_myproject/ap_start -into $blocksiggroup
add_wave /apatb_myproject_top/AESL_inst_myproject/ap_done -into $blocksiggroup
add_wave /apatb_myproject_top/AESL_inst_myproject/ap_idle -into $blocksiggroup
add_wave /apatb_myproject_top/AESL_inst_myproject/ap_ready -into $blocksiggroup
set resetgroup [add_wave_group "Reset" -into $designtopgroup]
add_wave /apatb_myproject_top/AESL_inst_myproject/ap_rst -into $resetgroup
set clockgroup [add_wave_group "Clock" -into $designtopgroup]
add_wave /apatb_myproject_top/AESL_inst_myproject/ap_clk -into $clockgroup
set testbenchgroup [add_wave_group "Test Bench Signals"]
set tbinternalsiggroup [add_wave_group "Internal Signals" -into $testbenchgroup]
set tb_simstatus_group [add_wave_group "Simulation Status" -into $tbinternalsiggroup]
set tb_portdepth_group [add_wave_group "Port Depth" -into $tbinternalsiggroup]
add_wave /apatb_myproject_top/AUTOTB_TRANSACTION_NUM -into $tb_simstatus_group -radix hex
add_wave /apatb_myproject_top/ready_cnt -into $tb_simstatus_group -radix hex
add_wave /apatb_myproject_top/done_cnt -into $tb_simstatus_group -radix hex
add_wave /apatb_myproject_top/LENGTH_data_0_V -into $tb_portdepth_group -radix hex
add_wave /apatb_myproject_top/LENGTH_data_1_V -into $tb_portdepth_group -radix hex
add_wave /apatb_myproject_top/LENGTH_data_2_V -into $tb_portdepth_group -radix hex
add_wave /apatb_myproject_top/LENGTH_data_3_V -into $tb_portdepth_group -radix hex
add_wave /apatb_myproject_top/LENGTH_data_4_V -into $tb_portdepth_group -radix hex
add_wave /apatb_myproject_top/LENGTH_data_5_V -into $tb_portdepth_group -radix hex
add_wave /apatb_myproject_top/LENGTH_data_6_V -into $tb_portdepth_group -radix hex
add_wave /apatb_myproject_top/LENGTH_data_7_V -into $tb_portdepth_group -radix hex
add_wave /apatb_myproject_top/LENGTH_data_8_V -into $tb_portdepth_group -radix hex
add_wave /apatb_myproject_top/LENGTH_data_9_V -into $tb_portdepth_group -radix hex
add_wave /apatb_myproject_top/LENGTH_res_0_V -into $tb_portdepth_group -radix hex
add_wave /apatb_myproject_top/LENGTH_const_size_in -into $tb_portdepth_group -radix hex
add_wave /apatb_myproject_top/LENGTH_const_size_out -into $tb_portdepth_group -radix hex
set tbcoutputgroup [add_wave_group "C Outputs" -into $testbenchgroup]
set tb_const_size_out_group [add_wave_group const_size_out(wire) -into $tbcoutputgroup]
add_wave /apatb_myproject_top/const_size_out_ap_vld -into $tb_const_size_out_group -color #ffff00 -radix hex
add_wave /apatb_myproject_top/const_size_out -into $tb_const_size_out_group -radix hex
set tb_const_size_in_group [add_wave_group const_size_in(wire) -into $tbcoutputgroup]
add_wave /apatb_myproject_top/const_size_in_ap_vld -into $tb_const_size_in_group -color #ffff00 -radix hex
add_wave /apatb_myproject_top/const_size_in -into $tb_const_size_in_group -radix hex
set tb_res_group [add_wave_group res(wire) -into $tbcoutputgroup]
add_wave /apatb_myproject_top/res_0_V_ap_vld -into $tb_res_group -color #ffff00 -radix hex
add_wave /apatb_myproject_top/res_0_V -into $tb_res_group -radix hex
set tbcinputgroup [add_wave_group "C Inputs" -into $testbenchgroup]
set tb_data_group [add_wave_group data(wire) -into $tbcinputgroup]
add_wave /apatb_myproject_top/data_9_V -into $tb_data_group -radix hex
add_wave /apatb_myproject_top/data_8_V -into $tb_data_group -radix hex
add_wave /apatb_myproject_top/data_7_V -into $tb_data_group -radix hex
add_wave /apatb_myproject_top/data_6_V -into $tb_data_group -radix hex
add_wave /apatb_myproject_top/data_5_V -into $tb_data_group -radix hex
add_wave /apatb_myproject_top/data_4_V -into $tb_data_group -radix hex
add_wave /apatb_myproject_top/data_3_V -into $tb_data_group -radix hex
add_wave /apatb_myproject_top/data_2_V -into $tb_data_group -radix hex
add_wave /apatb_myproject_top/data_1_V -into $tb_data_group -radix hex
add_wave /apatb_myproject_top/data_0_V -into $tb_data_group -radix hex
save_wave_config myproject.wcfg
run all
quit

