/*
 * Copyright (C) 2017 Pablo Correa Gómez
 *
 * GPLv3+
 *
 */

#include "small_conv.h"
#include "layers_def.h"
#include "caches.h"
#include "memory_controller.h"

void load_caches(data_t ifm_cache[K_SZ][X_PAR_UNROLL+2],
					memory_t input[MAX_INPUT_SIZE>>3][X_PAR_UNROLL],
					kernel_t weights_ker[TOTAL_WEIGHTS][9],
					kernel_t weights_bi[MAX_CH_OUT],
					kernel_t ker_cache[3][3],
					kernel_t* bias,
					layer_t layer,
					int col, int ch_in, int ch_out)
{

	//#pragma HLS DATAFLOW
	mem_ctr::calc_offsets(layer,col,ch_in,ch_out);

	caches::init_ifm(ifm_cache,input,layer,col);
	if (layer.type == CONV_3x3)
		caches::fetch_3x3_kernel_weights(ker_cache,weights_ker);
	else
		caches::fetch_1x1_kernel_weight(ker_cache,weights_ker);
	*bias = weights_bi[mem_ctr::current_offset_bias];

}

void fixed_col_conv(data_t ifm_cache[K_SZ][X_PAR_UNROLL+2],
					memory_t output[MAX_OUTPUT_SIZE >> 3][X_PAR_UNROLL],
					kernel_t kernel[K_SZ][K_SZ],
					kernel_t bias,
					layer_t layer,
					bool fire,
					int col, int ch_in, int ch_out)
{
	#pragma HLS INLINE

	#pragma HLS DEPENDENCE variable=output array inter RAW false
	#pragma HLS DEPENDENCE variable=output array inter WAW false

	product_data_t ofm_row_cache[X_PAR_UNROLL];
	#pragma HLS ARRAY_PARTITION variable=ofm_row_cache block factor=8 dim=1

	caches::ld_ofm_row(ofm_row_cache,output,bias,layer,col,ch_in);//Load a row from the output

	manual_partial_unroll: for (int sub_col = 0; sub_col < X_PAR_UNROLL; sub_col++) {
		#pragma HLS UNROLL

		if (layer.type == CONV_3x3) {
			slide_filter_ROWs: for (int ki = 0; ki < K_SZ; ki++){
				#pragma HLS UNROLL
				slide_filter_COLs: for (int kj = 0; kj < K_SZ; kj++){
					#pragma HLS UNROLL
					ofm_row_cache[sub_col] = ofm_row_cache[sub_col] + (ifm_cache[ki][sub_col+kj] * kernel[ki][kj]);
				}
			}
		} else {
			ofm_row_cache[sub_col] = ofm_row_cache[sub_col] + ifm_cache[1][sub_col+1] * kernel[1][1];
		}
	}
	caches::st_ofm_row(ofm_row_cache,output,layer,fire,col,ch_in); //Store current row value back to OFM
}

void conv2d(layer_t layer,
			memory_t input[MAX_INPUT_SIZE>>3][X_PAR_UNROLL],
			memory_t output[MAX_OUTPUT_SIZE >> 3][X_PAR_UNROLL],
			kernel_t weights_ker[TOTAL_WEIGHTS][9],
			kernel_t weights_bi[MAX_CH_OUT],
			bool fire,
			int ch_in, int ch_out)
{
	#pragma HLS INLINE

	data_t ifm_cache[K_SZ][X_PAR_UNROLL+2];
	#pragma HLS ARRAY_PARTITION variable=ifm_cache complete dim=0
	kernel_t ker_cache[3][3];
	#pragma HLS ARRAY_PARTITION variable=ker_cache complete dim=0
	kernel_t bias;

	read_img_pixels_X: for (int col = 0; col < (layer.in_pixel >> 3); col++){
		#pragma HLS LOOP_TRIPCOUNT min=1 max=16 avg=4 //Numbers divided by 8

		load_caches(ifm_cache,input,weights_ker,weights_bi,ker_cache,&bias,layer,col,ch_in,ch_out);

		read_img_pixels_Y: for (int row = 0; row < layer.in_pixel; row++){
			#pragma HLS LOOP_TRIPCOUNT min=8 max=128 avg=33
			#pragma HLS PIPELINE

			caches::load_ifm_row(ifm_cache,input,layer,row,col);
			if (layer.stride == 2 && (row & 1))
				continue;

			fixed_col_conv(ifm_cache,output,ker_cache,bias,layer,fire,col,ch_in,ch_out);
		}
	}
}

void hw_conv(layer_t layer,
		memory_t input[MAX_INPUT_SIZE>>3][X_PAR_UNROLL],
		memory_t output[MAX_OUTPUT_SIZE >> 3][X_PAR_UNROLL],
		kernel_t weights_ker[TOTAL_WEIGHTS][9],
		kernel_t weights_bi[MAX_CH_OUT],
		bool fire)
{
//mejor variable = fire
//#pragma HLS FUNCTION_INSTANTIATE variable=weights_ker

//#pragma HLS INLINE off
	IT_OUT_CH: for(int ch_out = 0; ch_out < layer.ch_out; ch_out++){
		#pragma HLS LOOP_TRIPCOUNT min=16 max=368 avg=133

		IT_IN_CH: for(int ch_in= 0; ch_in < layer.ch_in; ch_in++){
			#pragma HLS LOOP_TRIPCOUNT min=3 max=736 avg=149
			conv2d(layer,input,output,weights_ker,weights_bi,fire,ch_in,ch_out);
		}
	}
}
