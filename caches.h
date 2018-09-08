/*
 * Copyright (C) 2017 Pablo Correa Gómez
 *
 * GPLv3+
 *
 */

#ifndef __CACHES_H__
#define __CACHES_H__

#include "fpga_top.h"

namespace caches {
        void init_ifm(data_t ifm_cache[K_SZ][X_PAR_UNROLL+2],
                      memory_t input[MAX_INPUT_SIZE>>3][X_PAR_UNROLL],
                      layer_t layer,
                      int col);
        void load_ifm_row(data_t ifm_cache[K_SZ][X_PAR_UNROLL+2],
                          memory_t input[MAX_INPUT_SIZE>>3][X_PAR_UNROLL],
                          layer_t layer,
                          int row, int col);
        void ld_ofm_row(product_data_t ofm_row_cache[X_PAR_UNROLL],
                        memory_t output[MAX_OUTPUT_SIZE >> 3][X_PAR_UNROLL],
                        kernel_t bias,
                        layer_t layer,
                        int col, int ch_in);
        void st_ofm_row(product_data_t ofm_row_cache[X_PAR_UNROLL],
                        memory_t output[MAX_OUTPUT_SIZE >> 3][X_PAR_UNROLL],
                        layer_t layer,
                        bool fire,
                        int col, int ch_in);
        void fetch_3x3_kernel_weights(int8_t weights[3][3],
                                      kernel_t weights_ker[TOTAL_WEIGHTS][9]);
        void fetch_1x1_kernel_weight(int8_t weights[3][3],
                                     kernel_t weights_ker[TOTAL_WEIGHTS][9]);
}


#endif //__CACHES_H__
