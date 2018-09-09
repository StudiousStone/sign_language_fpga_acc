/*
 * Copyright (C) 2017 Pablo Correa Gómez
 *
 * GPLv3+
 *
 */

/*
 * This file contains the memory management. It makes sure that
 * the memory access is made transparent to the accelerator.
 * The values are store in the memories following the pattern:
 * input and output: ch-row-column, being ch the outtermost index.
 * kernel: layer-ch_out-ch_in-row-column.
 * In order to keep all the kernels aligned in multiples of 9, when
 * kernels are 1x1, it is expected that zero padding has been added.
 * In consequence the next layer's kernels can be read as if previous
 * kernels had been of size 3x3.
 */
#include "memory_controller.h"

// In order to speed up development and due to the big cost of ap_uint<>
// variables in software, these variables are avoided during simulation
#ifdef __SYNTHESIS__
//Output
ap_uint<20> mem_ctr::current_out_offset;
ap_uint<20> mem_ctr::current_ch_out_offset;
ap_uint<15> mem_ctr::offset_bt_ch_out;
ap_uint<8> mem_ctr::offset_bt_out_rows;

//Input
ap_uint<20> mem_ctr::current_in_offset;
ap_uint<20> mem_ctr::current_ch_in_offset;
ap_uint<15> mem_ctr::offset_bt_ch_in;
ap_uint<8> mem_ctr::offset_bt_in_rows;

//Bias
ap_uint<NBITS(TOTAL_BIAS)> mem_ctr::current_offset_bias; //TODO: Fix bit number

//Kernel
ap_uint<NBITS(TOTAL_WEIGHTS)> mem_ctr::current_offset_kernel;
ap_uint<NBITS(9)> mem_ctr::current_offset_1x1_kernel;

#else
//Output
uint32_t mem_ctr::current_out_offset;
uint32_t mem_ctr::current_ch_out_offset;
uint16_t mem_ctr::offset_bt_ch_out;
uint8_t mem_ctr::offset_bt_out_rows;

//Input
uint32_t mem_ctr::current_in_offset;
uint32_t mem_ctr::current_ch_in_offset;
uint16_t mem_ctr::offset_bt_ch_in;
uint8_t mem_ctr::offset_bt_in_rows;

//Bias
uint16_t mem_ctr::current_offset_bias;

//Kernel
uint32_t mem_ctr::current_offset_kernel;
uint8_t mem_ctr::current_offset_1x1_kernel;

#endif

//TODO: Should be placed inside the #ifdef
static ap_uint<NBITS(MAX_CH_OUT)> last_ch_in = 0;
static ap_uint<NBITS(MAX_CH_OUT)> last_ch_out = 0;

/*
 * Read input image from the input port and store it in memory
 */
void mem_ctr::init_mem_controller(data_t input_image[MAX_INPUT_SIZE],
                                  memory_t mem_i[MAX_INPUT_SIZE>>3][X_PAR_UNROLL])
{
        int cnt = 0;
        for (int ch_in = 0; ch_in < IM_CH_IN; ch_in++) {
                for (int row = 0; row < IM_PX_SZ; row++) {
                        for (int col = 0; col < IM_PX_SZ >> 3; col++) {
                                for (int sub_col = 0; sub_col < X_PAR_UNROLL; sub_col++) {
                                        mem_i[cnt][sub_col] = input_image[cnt * X_PAR_UNROLL + sub_col];
                                }
                                cnt++;
                        }
                }
        }
}

/*
 * Initialize memory offsets for the first layer
 */
void mem_ctr::init_offsets(layer_t layer)
{
        last_ch_in = 0;
        last_ch_out = 0;

        current_in_offset = 0;
        current_ch_in_offset =
        offset_bt_in_rows = layer.in_pixel >> 3;
        offset_bt_ch_in = layer.in_pixel*layer.in_pixel >> 3;

        current_out_offset = 0;
        current_ch_out_offset = 0;
        offset_bt_out_rows = layer.out_pixel >> 3;
        offset_bt_ch_out = layer.out_pixel*layer.out_pixel >> 3;

        current_offset_bias = 0;
        current_offset_kernel = 0;
        current_offset_1x1_kernel = 0;
}

/*
 * Prepare offsets for the layer's execution
 * If it is a concatenation layer, it has to be taken into account
 */
void mem_ctr::set_offsets(layer_t layer, bool concat)
{
        last_ch_in = 0;
        last_ch_out = 0;

        current_in_offset = 0;
        current_ch_in_offset = 0;
        offset_bt_in_rows = layer.in_pixel >> 3;
        offset_bt_ch_in = layer.in_pixel*layer.in_pixel >> 3;

        if (concat) {
                current_ch_out_offset += offset_bt_ch_out;
        } else {
                current_ch_out_offset = 0;
                offset_bt_out_rows = layer.out_pixel >> 3;
                offset_bt_ch_out = layer.out_pixel*layer.out_pixel >> 3;
        }
        current_out_offset = 0;

        current_offset_kernel++;
        current_offset_1x1_kernel = 0;
        current_offset_bias++;
}

/*
 * For the execution of every new (sub)column, the offsets have to
 * be updated in order for the access methods to follow the information
 * storage's pattern in memories.
 */
void mem_ctr::calc_offsets(layer_t layer,
                           int col, int ch_in, int ch_out)
{
#pragma HLS INLINE

        if (last_ch_in != ch_in) {
                if (last_ch_out != ch_out) {
                        current_ch_in_offset = 0;
                        current_ch_out_offset += offset_bt_ch_out;
                        current_offset_bias++;
                } else {
                        current_ch_in_offset += offset_bt_ch_in;
                }
        }

        if (layer.type == CONV_3x3 && last_ch_in != ch_in) {
                current_offset_kernel++;
        } else {
                if (last_ch_in != ch_in) {
                        if (current_offset_1x1_kernel >= 8) {
                                current_offset_1x1_kernel = 0;
                                current_offset_kernel++;
                        } else {
                                current_offset_1x1_kernel++;
                        }
                }
        }

        last_ch_out = ch_out;
        last_ch_in = ch_in;

        if (layer.stride == 2) {
                current_out_offset = current_ch_out_offset + (col >> 1);

        } else {
                current_out_offset = current_ch_out_offset + col;
        }


        current_in_offset = current_ch_in_offset;
}
