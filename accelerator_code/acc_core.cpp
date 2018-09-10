/*
 * Copyright (C) 2017 Pablo Correa Gómez
 *
 * GPLv3+
 *
 */
/*
 *  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 *  |1| | | | | | | |1|2|3|4|5|6|7|8|
 *  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 *   ^              |               |
 *   |              |    subcol     |
 *  col             |               |
 *
 */
#include "acc_core.h"
#include "layers_def.h"
#include "caches.h"
#include "memory_controller.h"

void load_caches(data_t ifm_cache[K_SZ][X_PAR_UNROLL+2],
                 memory_t input[MAX_INPUT_SIZE >> 3][X_PAR_UNROLL],
                 kernel_t weights_ker[TOTAL_WEIGHTS][9],
                 kernel_t weights_bi[MAX_CH_OUT],
                 kernel_t ker_cache[3][3],
                 kernel_t* bias,
                 layer_t layer,
                 int sub_col)
{
        //#pragma HLS DATAFLOW

        caches::init_ifm(ifm_cache,input,layer,sub_col);
        if (layer.type == CONV_3x3)
                caches::fetch_3x3_kernel_weights(ker_cache,weights_ker);
        else
                caches::fetch_1x1_kernel_weight(ker_cache,weights_ker);
        *bias = weights_bi[mem_ctr::current_offset_bias];

}

void conv_3x3(data_t ifm_cache[K_SZ][X_PAR_UNROLL+2],
              product_data_t ofm_cache[X_PAR_UNROLL],
              kernel_t kernel[K_SZ][K_SZ],
              int col)
{
        #pragma HLS INLINE

 slide_filter_ROWs:
        for (int ki = 0; ki < K_SZ; ki++){
                #pragma HLS UNROLL
        slide_filter_COLs:
                for (int kj = 0; kj < K_SZ; kj++){
                        #pragma HLS UNROLL
                        ofm_cache[col] += (ifm_cache[ki][col+kj] * kernel[ki][kj]);
                }
        }
}

void fixed_sub_col_conv(data_t ifm_cache[K_SZ][X_PAR_UNROLL+2],
                        product_data_t ofm_cache[X_PAR_UNROLL],
                        kernel_t kernel[K_SZ][K_SZ],
                        layer_t layer)
{
        #pragma HLS INLINE

 it_row_elements_of_subcol:
        for (int col = 0; col < X_PAR_UNROLL; col++) {
                #pragma HLS UNROLL

                if (layer.type == CONV_3x3) {
                        conv_3x3(ifm_cache,ofm_cache,kernel,col);
                } else {
                        ofm_cache[col] += ifm_cache[1][col+1] * kernel[1][1];
                }
        }
}

void conv2d(layer_t layer,
            memory_t input[MAX_INPUT_SIZE >> 3][X_PAR_UNROLL],
            memory_t output[MAX_OUTPUT_SIZE >> 3][X_PAR_UNROLL],
            kernel_t weights_ker[TOTAL_WEIGHTS][9],
            kernel_t weights_bi[MAX_CH_OUT],
            int ch_in, int ch_out)
{
        #pragma HLS INLINE

        data_t ifm_cache[K_SZ][X_PAR_UNROLL+2];
        product_data_t ofm_cache[X_PAR_UNROLL];
        kernel_t ker_cache[3][3];
        #pragma HLS ARRAY_PARTITION variable=ifm_cache complete dim=0
        #pragma HLS ARRAY_PARTITION variable=ofm_cache block factor=8 dim=1
        #pragma HLS ARRAY_PARTITION variable=ker_cache complete dim=0
        kernel_t bias;

 it_subcols_size_8:
        for (int sub_col = 0; sub_col < (layer.in_pixel >> 3); sub_col++){
                #pragma HLS LOOP_TRIPCOUNT min=1 max=16 avg=4 //Numbers divided by 8

                mem_ctr::calc_offsets(layer,sub_col,ch_in,ch_out);
                load_caches(ifm_cache,input,weights_ker,weights_bi,ker_cache,&bias,layer,sub_col);

        it_rows_of_subcol:
                for (int row = 0; row < layer.in_pixel; row++){
                        #pragma HLS LOOP_TRIPCOUNT min=8 max=128 avg=33
                        #pragma HLS PIPELINE

                        caches::load_ifm_row(ifm_cache,input,layer,row,sub_col);
                        if (layer.stride == 2 && (row & 1))
                                continue;

                        //TODO: Check how is this gonna work...
                        #pragma HLS DEPENDENCE variable=output array inter RAW false
                        #pragma HLS DEPENDENCE variable=output array inter WAW false


                        caches::ld_ofm_row(ofm_cache,output,bias,layer,sub_col,ch_in);
                        fixed_sub_col_conv(ifm_cache,ofm_cache,ker_cache,layer);
                        caches::st_ofm_row(ofm_cache,output,layer,sub_col,ch_in);
                }
        }
}

void hw_conv(layer_t layer,
             memory_t input[MAX_INPUT_SIZE >> 3][X_PAR_UNROLL],
             memory_t output[MAX_OUTPUT_SIZE >> 3][X_PAR_UNROLL],
             kernel_t weights_ker[TOTAL_WEIGHTS][9],
             kernel_t weights_bi[MAX_CH_OUT],
             bool fire)
{
        //mejor variable = fire
        #pragma HLS FUNCTION_INSTANTIATE variable=fire

 out_ch:
        for(int ch_out = 0; ch_out < layer.ch_out; ch_out++){
                #pragma HLS LOOP_TRIPCOUNT min=16 max=368 avg=133

        in_ch:
                for(int ch_in= 0; ch_in < layer.ch_in; ch_in++){
                        #pragma HLS LOOP_TRIPCOUNT min=3 max=736 avg=149
                        conv2d(layer,input,output,weights_ker,weights_bi,ch_in,ch_out);
                }
        }
}
