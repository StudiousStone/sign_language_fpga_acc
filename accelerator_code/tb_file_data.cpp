#include <stdio.h>
#include "fpga_top.h"

#if 1
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
#endif
