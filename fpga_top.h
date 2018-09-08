/*
 * Copyright (C) 2017 Pablo Correa Gómez
 *
 * GPLv3+
 *
 */

#ifndef __FPGA_TOP_H__
#define __FPGA_TOP_H__

#include "layers_def.h"
#include <stdint.h>


// NBITS(constant) = how many bits needed to represent <constant>
#define NBITS2(n) ((n & 2) ? 1 : 0)
#define NBITS4(n) ((n & (0xC)) ? (2 + NBITS2(n >> 2)) : (NBITS2(n)))
#define NBITS8(n) ((n & 0xF0) ? (4 + NBITS4(n >> 4)) : (NBITS4(n)))
#define NBITS16(n) ((n & 0xFF00) ? (8 + NBITS8(n >> 8)) : (NBITS8(n)))
#define NBITS32(n) ((n & 0xFFFF0000) ? (16 + NBITS16(n >> 16)) : (NBITS16(n)))
#define NBITS(n) ((n) == 0 ? 1 : NBITS32((n)) + 1)

#define KERNEL_SIZE 3
#define K_SZ KERNEL_SIZE
#define PAD 1

#define X_PAR_UNROLL 8

typedef int32_t product_data_t;
//Values involved in the interface have to be regular types
typedef int8_t kernel_t;
typedef int8_t data_t;
typedef int16_t memory_t;
typedef uint8_t result_t;

result_t fpga_top(data_t image[MAX_INPUT_SIZE],
                  bool load,
                  kernel_t bias[MAX_CH_OUT],
                  kernel_t kernels[TOTAL_WEIGHTS][9]);

#endif //__FPGA_TOP_H__
