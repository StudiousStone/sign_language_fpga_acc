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
                        int col, int ch_in);
        void fetch_3x3_kernel_weights(int8_t weights[3][3],
                                      kernel_t weights_ker[TOTAL_WEIGHTS][9]);
        void fetch_1x1_kernel_weight(int8_t weights[3][3],
                                     kernel_t weights_ker[TOTAL_WEIGHTS][9]);
}


#endif //__CACHES_H__
