/*
 * Copyright (C) 2017 Pablo Correa Gómez
 *
 * GPLv3+
 *
 */

#ifndef __LAYERS_DEF_H__
#define __LAYERS_DEF_H__

#include <stdint.h>
#include <ap_int.h>
#include "layer_definition.h"

//IFM_SIZE = (((OFM_SIZE-1)*STR) + KERNEL_SIZE - (2*PAD))


struct Operation {
#ifdef __SYNTHESIS__
        int8_t layer;
        int8_t type; // 0 = conv1x1; 1 = conv_3x3
        ap_uint<2> stride;
        ap_uint<10> ch_in;
        ap_uint<9> ch_out;
        ap_uint<9> in_pixel;
        ap_uint<8> out_pixel;
        ap_int<5> frac_in;
        ap_int<5> frac_out;
        ap_int<5> frac_wei;
#else
        int8_t layer;
        int8_t type; // 0 = conv1x1; 1 = conv_3x3
        int8_t stride;
        uint16_t ch_in;
        uint16_t ch_out;
        uint16_t in_pixel;
        uint8_t out_pixel;
        int8_t frac_in;
        int8_t frac_out;
        int8_t frac_wei;
#endif
};

typedef const struct Operation layer_t;

#endif //__LAYERS_DEF_H__
