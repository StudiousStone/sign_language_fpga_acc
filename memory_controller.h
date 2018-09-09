/*
 * Copyright (C) 2017 Pablo Correa Gómez
 *
 * GPLv3+
 *
 */

#ifndef __MEMORY_CONTROLLER_H__
#define __MEMORY_CONTROLLER_H__

#include "fpga_top.h"

namespace mem_ctr {

// In order to speed up development and due to the big cost of ap_uint<>
// variables in software, these variables are avoided during simulation
#ifdef __SYNTHESIS__
        //Output controller
        extern ap_uint<20> current_out_offset;
        extern ap_uint<20> current_ch_out_offset;
        extern ap_uint<15> offset_bt_ch_out;
        extern ap_uint<8> offset_bt_out_rows;

        //Input controller
        extern ap_uint<20> current_in_offset;
        extern ap_uint<20> current_ch_in_offset;
        extern ap_uint<15> offset_bt_ch_in;
        extern ap_uint<8> offset_bt_in_rows;

        //Bias controller
        extern ap_uint<NBITS(TOTAL_BIAS)> current_offset_bias; //TODO: Fix bit number

        //Kernel controller
        extern ap_uint<NBITS(TOTAL_WEIGHTS)> current_offset_kernel;
        extern ap_uint<NBITS(9)> current_offset_1x1_kernel;
#else
        //Output controller
        extern uint32_t current_out_offset;
        extern uint32_t current_ch_out_offset;
        extern uint16_t offset_bt_ch_out;
        extern uint8_t offset_bt_out_rows;

        //Input controller
        extern uint32_t current_in_offset;
        extern uint32_t current_ch_in_offset;
        extern uint16_t offset_bt_ch_in;
        extern uint8_t offset_bt_in_rows;

        //Bias controller
        extern uint16_t current_offset_bias;

        //Kernel controller
        extern uint32_t current_offset_kernel;
        extern uint8_t current_offset_1x1_kernel;

#endif

        void init_mem_controller(data_t input_image[MAX_INPUT_SIZE],
                                 memory_t mem_i[MAX_INPUT_SIZE>>3][X_PAR_UNROLL]);
        void config_controller(layer_t layer);
        void set_offsets_next_lay(bool concat);
        void calc_offsets(layer_t layer,
                          int col, int ch_in, int ch_out);
}


#endif //__MEMORY_CONTROLLER_H__
