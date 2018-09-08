/*
 * Copyright (C) 2017 Pablo Correa Gómez
 *
 * GPLv3+
 *
 */

#ifndef __SMALL_CONV_H__
#define __SMALL_CONV_H__


#include "fpga_top.h"

//#define O_MAP(c_o,h,w) (((c_o)*OUT_PX*OUT_PX) + ((h)*OUT_PX) + (w))

void hw_conv(layer_t layer,
             memory_t input[MAX_INPUT_SIZE >> 3][X_PAR_UNROLL],
             memory_t output[MAX_OUTPUT_SIZE >> 3][X_PAR_UNROLL],
             kernel_t weights_ker[TOTAL_WEIGHTS][9],
             kernel_t weights_bi[MAX_CH_OUT],
             bool fire
             );

#endif //__SMALL_CONV_H__
