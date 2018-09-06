/*
 * Copyright (C) 2017 Pablo Correa Gómez
 *
 * GPLv3+
 *
 */

#include "caches.h"
#include "memory_controller.h"

void caches::init_ifm(data_t ifm_cache[K_SZ][X_PAR_UNROLL+2],
						memory_t input[MAX_INPUT_SIZE >> 3][X_PAR_UNROLL],
						layer_t layer, int col)
{
	#pragma HLS PIPELINE
	#pragma HLS INLINE

	caches::load_ifm_row(ifm_cache,input,layer,-2,col);
	caches::load_ifm_row(ifm_cache,input,layer,-1,col);

}

void caches::load_ifm_row(data_t ifm_cache[K_SZ][X_PAR_UNROLL+2],
							memory_t input[MAX_INPUT_SIZE >> 3][X_PAR_UNROLL],
							layer_t layer,
							int row, int col)
{
	int16_t idx, idx_last, idx_first;
	int16_t idy = row + 2 - PAD;
	#pragma HLS INLINE
	#pragma HLS PIPELINE
	load_row: for (int j = 1; j < X_PAR_UNROLL+1; j++) {
		#pragma HLS UNROLL
		idx = X_PAR_UNROLL*col + j - PAD;

		ifm_cache[0][j] = ifm_cache[1][j];
		ifm_cache[1][j] = ifm_cache[2][j];
		if (idy >= layer.in_pixel || idy < 0)
			ifm_cache[2][j] = 0;
		else
			ifm_cache[2][j] = input[mem_ctr::current_in_offset + col][j - PAD];
	}

	ifm_cache[0][0] = ifm_cache[1][0];
	ifm_cache[1][0] = ifm_cache[2][0];
	idx_first = X_PAR_UNROLL*col - PAD;
	if (idx_first < 0 || idy >= layer.in_pixel || idy < 0)
		ifm_cache[2][0] = 0;
	else
		ifm_cache[2][0] = input[mem_ctr::current_in_offset + col - 1][X_PAR_UNROLL-1];

	ifm_cache[0][X_PAR_UNROLL + 1] = ifm_cache[1][X_PAR_UNROLL + 1];
	ifm_cache[1][X_PAR_UNROLL + 1] = ifm_cache[2][X_PAR_UNROLL + 1];
	idx_last = X_PAR_UNROLL*col + X_PAR_UNROLL + 1 - PAD;
	if (idx_last >= layer.in_pixel || idy >= layer.in_pixel || idy < 0)
		ifm_cache[2][X_PAR_UNROLL+1] = 0;
	else
		ifm_cache[2][X_PAR_UNROLL+1] = input[mem_ctr::current_in_offset + col + 1][0];

	//Increase the offset if the row is not a padding row
	if (!(idy >= layer.in_pixel || idy < 0))
		mem_ctr::current_in_offset += mem_ctr::offset_bt_in_rows;
}

void caches::fetch_3x3_kernel_weights(kernel_t weights[3][3],
									kernel_t weights_ker[TOTAL_WEIGHTS][9])
{
	fetch_ker_row: for (int i = 0; i < 3; i++) {
		#pragma HLS UNROLL
		fetch_ker_col: for (int j = 0; j < 3; j++) {
			#pragma HLS UNROLL
			weights[i][j] = weights_ker[mem_ctr::current_offset_kernel][i * 3 + j];
		}
	}
}

void caches::fetch_1x1_kernel_weight(int8_t weights[3][3],
							kernel_t weights_ker[TOTAL_WEIGHTS][9])
{
	weights[1][1] = weights_ker[mem_ctr::current_offset_kernel][mem_ctr::current_offset_1x1_kernel];
}

void caches::ld_ofm_row(product_data_t ofm_row_cache[X_PAR_UNROLL],
						memory_t output[MAX_OUTPUT_SIZE >> 3][X_PAR_UNROLL],
						kernel_t bias,
						layer_t layer,
						int col, int ch_in)
{
	#pragma HLS INLINE

	int sub_col_aux;
	const int fixed_slide = layer.frac_in + layer.frac_wei - layer.frac_out;
	const int bias_slide = layer.frac_in;
	int32_t bias_cache;
	if (bias_slide > 0)
		bias_cache = bias << bias_slide;
	else
		bias_cache = bias >> (-bias_slide);

	ld_ofm_row: for (int sub_col = 0; sub_col < X_PAR_UNROLL; sub_col++) {
		#pragma HLS UNROLL
		sub_col_aux = sub_col;

		if (layer.stride == 2) {
			if (sub_col & 1)
				continue;

			sub_col_aux = sub_col_aux >> 1;
			if (col & 1)
				sub_col_aux += 4;
		}

		if (ch_in)
			ofm_row_cache[sub_col] = ((int32_t) output[mem_ctr::current_out_offset][sub_col_aux]) << fixed_slide;
		else
			ofm_row_cache[sub_col] = bias_cache;

	}
}

void relu(bool fire,
		product_data_t output_cache[X_PAR_UNROLL],
		int sub_col,
		layer_t layer)
{
	#pragma HLS INLINE
	if (layer.layer == TOTAL_OPS) {
		if (output_cache[sub_col] < INT8_MIN)
			output_cache[sub_col] = INT8_MIN;
	} else {
		if (output_cache[sub_col] < 0)
			output_cache[sub_col] = 0;
	}

}


void caches::st_ofm_row(product_data_t ofm_row_cache[X_PAR_UNROLL],
						memory_t output[MAX_OUTPUT_SIZE >> 3][X_PAR_UNROLL],
						layer_t layer,
						bool fire,
						int col, int ch_in)
{
	#pragma HLS INLINE

	const int fixed_slide = layer.frac_in + layer.frac_wei - layer.frac_out;
	product_data_t output_cache[X_PAR_UNROLL];
	#pragma HLS ARRAY_PARTITION variable=output_cache complete factor=8 dim=1


	if (layer.stride == 2) {
		int sub_col_aux;

		if (col & 1)
			sub_col_aux = 4;
		else
			sub_col_aux = 0;

		st_ofm_row_stride: for (int sub_col = 0; sub_col < (X_PAR_UNROLL>>1); sub_col++) {
			#pragma HLS UNROLL

			output_cache[sub_col] = ofm_row_cache[sub_col*2] + (1 << (fixed_slide - 1));
			output_cache[sub_col] = output_cache[sub_col] >> fixed_slide;

			if (ch_in == layer.ch_in-1) {
				relu(fire,output_cache,sub_col,layer);

				if (output_cache[sub_col] > INT8_MAX)
					output_cache[sub_col] = INT8_MAX;
			}

			if (output_cache[sub_col] >= INT16_MAX)
				output_cache[sub_col] = INT16_MAX;

			output[mem_ctr::current_out_offset][sub_col_aux+sub_col] = output_cache[sub_col];
		}
	} else {
		st_ofm_row_: for (int sub_col = 0; sub_col < X_PAR_UNROLL; sub_col++) {
			#pragma HLS UNROLL

			output_cache[sub_col] = ofm_row_cache[sub_col] + (1 << (fixed_slide - 1));
			output_cache[sub_col] = output_cache[sub_col] >> fixed_slide;

			if (ch_in == layer.ch_in-1) {
				relu(fire,output_cache,sub_col,layer);

				if (output_cache[sub_col] >= INT8_MAX)
					output_cache[sub_col] = INT8_MAX;
			}

			if (output_cache[sub_col] >= INT16_MAX)
				output_cache[sub_col] = INT16_MAX;
			output[mem_ctr::current_out_offset][sub_col] =  output_cache[sub_col];
		}
	}
	mem_ctr::current_out_offset += mem_ctr::offset_bt_out_rows;
}

