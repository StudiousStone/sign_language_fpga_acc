/*****************************************************************************
* Copyright (C) 2018 Pablo Correa Gomez                                      *
*                                                                            *
*    This program is free software; you can redistribute it and/or modify    *
*    it under the terms of the GNU Affero General Public License as          *
*    published by the Free Software Foundation; either version 3 of          *
*    the License, or (at your option) any later version.                     *
*                                                                            *
*    This program is distributed in the hope that it will be useful,         *
*    but WITHOUT ANY WARRANTY; without even the implied warranty of          *
*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           *
*    GNU General Public License for more details.                            *
*    ( http://www.fsf.org/licenses/agpl.txt )                                *
*****************************************************************************/

#include <stdio.h>
#include "fpga_top.h"

int main(){
        int result, errorn;
        const int golden_result = 1;

        static data_t input[MAX_INPUT_SIZE] = {
#include "goldendata/B_1.txt"
        };

        static kernel_t bias[TOTAL_BIAS] = {
#include "goldendata/bias_all.txt"
        };
        static kernel_t kernels[TOTAL_WEIGHTS][9];

        int ker_val;
        FILE *f_ker = fopen("goldendata/kernel_all.txt","r");
        if (f_ker == NULL) {
                printf("*** Error opening kernels file");
                return 1;
        }
        for (int i = 0; i < TOTAL_WEIGHTS; i++) {
                for (int j = 0; j < 9; j++) {
                        fscanf(f_ker,"%i,",&ker_val);
                        kernels[i][j] = (kernel_t) ker_val;
                }
        }
        fclose(f_ker);

        errorn = fpga_top(input,true,bias,kernels);
        if (errorn) {
                printf("*** Error loading weights ***");
                return 1;
        }

        result = fpga_top(input,false,bias,kernels);

        if (result != golden_result) {
                printf("**** ERROR: Wrong prediction ****\n");
                printf("**** hw_result: %i, golden_result: %i ****\n",result,golden_result);
                return 1;
        } else {
                printf("**** Right prediction, %i ****\n",golden_result);
        }

        return 0;
}
