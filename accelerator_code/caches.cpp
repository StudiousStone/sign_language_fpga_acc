/************************************************************************************
* Copyright (C) 2018 Pablo Correa Gomez										        *
*                                                                                   *
*    This program is free software; you can redistribute it and/or modify           *
*    it under the terms of the GNU Affero General Public License as                 *
*    published by the Free Software Foundation; either version 3 of                 *
*    the License, or (at your option) any later version.                            *
*                                                                                   *
*    This program is distributed in the hope that it will be useful,                *
*    but WITHOUT ANY WARRANTY; without even the implied warranty of                 *
*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                  *
*    GNU General Public License for more details.                                   *
*    ( http://www.fsf.org/licenses/agpl.txt )                                       *
************************************************************************************/

#include "caches.h"
#include "memory_controller.h"

/*
 *
 *
 *
 */
void caches::init_ifm(data_t ifm_cache[K_SZ][X_PAR_UNROLL+2],
                      memory_t input[MAX_INPUT_SIZE >> 3][X_PAR_UNROLL],
                      layer_t layer, int sub_col)
{
#pragma HLS PIPELINE
#pragma HLS INLINE

        caches::load_ifm_row(ifm_cache,input,layer,-2,sub_col);
        caches::load_ifm_row(ifm_cache,input,layer,-1,sub_col);

}

/*
 * Load one row of input from memory to cache.
 * The elements currently stored in the cache are sent to the "upper row" register
 * The 8 central elements have a fixed pattern to be loaded,
 * while the leftmost and rightmost elements have to be checked independently
 */
void caches::load_ifm_row(data_t ifm_cache[K_SZ][X_PAR_UNROLL+2],
                          memory_t input[MAX_INPUT_SIZE >> 3][X_PAR_UNROLL],
                          layer_t layer,
                          int row, int sub_col)
{
        int16_t idx, idx_last, idx_first;
        int16_t idy = row + 2 - PAD;
#pragma HLS INLINE
#pragma HLS PIPELINE
        // 8 central elements
 load_row:
        for (int j = 1; j < X_PAR_UNROLL+1; j++) {
#pragma HLS UNROLL
                idx = X_PAR_UNROLL*sub_col + j - PAD;

                ifm_cache[0][j] = ifm_cache[1][j];
                ifm_cache[1][j] = ifm_cache[2][j];
                if (idy >= layer.in_pixel || idy < 0)
                        ifm_cache[2][j] = 0;
                else
                        ifm_cache[2][j] = input[mem_ctr::current_in_offset + sub_col][j - PAD];
        }

        // Left most element
        ifm_cache[0][0] = ifm_cache[1][0];
        ifm_cache[1][0] = ifm_cache[2][0];
        idx_first = X_PAR_UNROLL*sub_col - PAD;
        if (idx_first < 0 || idy >= layer.in_pixel || idy < 0)
                ifm_cache[2][0] = 0;
        else
                ifm_cache[2][0] = input[mem_ctr::current_in_offset + sub_col - 1][X_PAR_UNROLL-1];

        // Right most element
        ifm_cache[0][X_PAR_UNROLL + 1] = ifm_cache[1][X_PAR_UNROLL + 1];
        ifm_cache[1][X_PAR_UNROLL + 1] = ifm_cache[2][X_PAR_UNROLL + 1];
        idx_last = X_PAR_UNROLL*sub_col + X_PAR_UNROLL + 1 - PAD;
        if (idx_last >= layer.in_pixel || idy >= layer.in_pixel || idy < 0)
                ifm_cache[2][X_PAR_UNROLL+1] = 0;
        else
                ifm_cache[2][X_PAR_UNROLL+1] = input[mem_ctr::current_in_offset + sub_col + 1][0];

        //Increase the offset if the row is not a padding row
        if (!(idy >= layer.in_pixel || idy < 0))
                mem_ctr::current_in_offset += mem_ctr::offset_bt_in_rows;
}

/*
 * Load 3x3 kernel from memory to cache
 */
void caches::fetch_3x3_kernel_weights(kernel_t weights[3][3],
                                      kernel_t weights_ker[TOTAL_WEIGHTS][9])
{
 fetch_ker_row:
        for (int i = 0; i < 3; i++) {
#pragma HLS UNROLL
        fetch_ker_col:
                for (int j = 0; j < 3; j++) {
#pragma HLS UNROLL
                        weights[i][j] = weights_ker[mem_ctr::current_offset_kernel][i * 3 + j];
                }
        }
}

/*
 * Load 1x1 kernel from memory to central element of cache
 */
void caches::fetch_1x1_kernel_weight(int8_t weights[3][3],
                                     kernel_t weights_ker[TOTAL_WEIGHTS][9])
{
        weights[1][1] = weights_ker[mem_ctr::current_offset_kernel][mem_ctr::current_offset_1x1_kernel];
}

/*
 * Load one output row from memory to cache.
 * First time it is initialized, the bias instead of the output memory is read
 * The different precision between the output memory and the cache has to be considered
 * For layers with stride 2, the memory pattern is different in the memories
 * and the caches, so it has to be taken in consideration
 */
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

 ld_ofm_row:
        for (int sub_col = 0; sub_col < X_PAR_UNROLL; sub_col++) {
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

static void relu(product_data_t output_cache[X_PAR_UNROLL],
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

/*
 * Store one output row from memory to cache.
 * The last time a value has to be stored, a relu is applied.
 * The different precision between the output memory and the cache
 * as well as possible overflows when storing the data have to be considered.
 * For layers with stride 2, the memory pattern is different in the memories
 * and the caches, so it also has to be taken in consideration
 */
void caches::st_ofm_row(product_data_t ofm_row_cache[X_PAR_UNROLL],
                        memory_t output[MAX_OUTPUT_SIZE >> 3][X_PAR_UNROLL],
                        layer_t layer,
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

        st_ofm_row_stride_2:
                for (int sub_col = 0; sub_col < (X_PAR_UNROLL>>1); sub_col++) {
#pragma HLS UNROLL

                        output_cache[sub_col] = ofm_row_cache[sub_col*2] + (1 << (fixed_slide - 1));
                        output_cache[sub_col] = output_cache[sub_col] >> fixed_slide;

                        if (ch_in == layer.ch_in-1) {
                                relu(output_cache,sub_col,layer);

                                if (output_cache[sub_col] > INT8_MAX)
                                        output_cache[sub_col] = INT8_MAX;
                        }

                        if (output_cache[sub_col] >= INT16_MAX)
                                output_cache[sub_col] = INT16_MAX;

                        output[mem_ctr::current_out_offset][sub_col_aux+sub_col] = output_cache[sub_col];
                }
        } else {
        st_ofm_row_stride_1:
                for (int sub_col = 0; sub_col < X_PAR_UNROLL; sub_col++) {
#pragma HLS UNROLL

                        output_cache[sub_col] = ofm_row_cache[sub_col] + (1 << (fixed_slide - 1));
                        output_cache[sub_col] = output_cache[sub_col] >> fixed_slide;

                        if (ch_in == layer.ch_in-1) {
                                relu(output_cache,sub_col,layer);

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
