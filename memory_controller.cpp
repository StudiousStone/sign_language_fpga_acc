/*
 * Copyright (C) 2017 Pablo Correa Gómez
 *
 * GPLv3+
 *
 */

#include "memory_controller.h"

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

static ap_uint<NBITS(MAX_CH_OUT)> last_ch_in = 0;
static ap_uint<NBITS(MAX_CH_OUT)> last_ch_out = 0;

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

	current_out_offset = 0;
	current_ch_out_offset = 0;

	current_in_offset = 0;
	current_ch_in_offset = 0;

	current_offset_bias = 0;
	current_offset_kernel = 0;
}

void mem_ctr::config_controller(layer_t layer)
{
	last_ch_in = 0;
	last_ch_out = 0;

	offset_bt_out_rows = layer.out_pixel >> 3;
	offset_bt_ch_out = layer.out_pixel*layer.out_pixel >> 3;

	offset_bt_in_rows = layer.in_pixel >> 3;
	offset_bt_ch_in = layer.in_pixel*layer.in_pixel >> 3;
}

void mem_ctr::set_offsets_next_lay(bool concat)
{
	current_in_offset = 0;
	current_ch_in_offset = 0;

	if (concat)
		current_ch_out_offset += offset_bt_ch_out;
	else
		current_ch_out_offset = 0;
	current_out_offset = 0;

	current_offset_kernel++;
	current_offset_1x1_kernel = 0;
	current_offset_bias++;
}

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
