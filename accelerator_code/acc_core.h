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

#ifndef __ACC_CORE__
#define __ACC_CORE__


#include "fpga_top.h"

//#define O_MAP(c_o,h,w) (((c_o)*OUT_PX*OUT_PX) + ((h)*OUT_PX) + (w))

void hw_conv(layer_t layer,
             memory_t input[MAX_INPUT_SIZE >> 3][X_PAR_UNROLL],
             memory_t output[MAX_OUTPUT_SIZE >> 3][X_PAR_UNROLL],
             kernel_t weights_ker[TOTAL_WEIGHTS][9],
             kernel_t weights_bi[MAX_CH_OUT],
             bool fire
             );

#endif //__ACC_CORE__
