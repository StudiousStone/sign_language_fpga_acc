/*
 * Copyright (C) 2017 Pablo Correa Gómez
 *
 * GPLv3+
 *
 */
#include "fpga_top.h"
#include <stdio.h>

#if 0
void rm_nl(char *str)
{
	int i = 0;
	while (str[i]) {
		i++;
	}

	if (str[i - 1] == '\n') {
		str[i - 1] = '\0';
	}

	return;
}

double get_output_statistics(int golden_result,int hw_result,const char *input_name)
{
	static int accuracy;
	static int image_num;

	if (hw_result != golden_result) {
	   printf("**** Wrong prediction for file %s ****\n",input_name);
	   printf("**** hw_result: %c, golden_result: %c ****\n",hw_result + 'A',golden_result + 'A');
	} else {
		printf("**** Right prediction for file %s ****\n",input_name);
		accuracy++;
	}

	image_num++;
	return ((double) accuracy) / image_num;
}

int read_in_data_and_golden_res(FILE *f_input,data_t input[MAX_INPUT_SIZE],
			FILE *f_results)
{
	int input_data,golden_result;
	//Get input data
	for (int i = 0; i < MAX_INPUT_SIZE; i++) {
		fscanf(f_input,"%i,",&input_data);
		input[i] = input_data;
	}

	//Get golden data
	fscanf(f_results,"%i,",&golden_result);
	return golden_result;
}

int main()
{

#define F_NAME_SIZE 128
#define MIN_ACCURACY 0.78
#define READ_FILE

	int errorn = 0;
	static data_t input[MAX_INPUT_SIZE];
	int golden_result, hw_result;
	double accuracy;

	static kernel_t bias[TOTAL_BIAS] = {
		#include "goldendata/bias_all.txt"
	};
#ifndef READ_FILE
	static kernel_t kernels[TOTAL_WEIGHTS][9] = {
		#include "goldendata/kernel_all.txt"
	};
#else
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
#endif

	//Load weights in FPGA
	errorn = fpga_top(input,true,bias,kernels);
	if (errorn) {
		printf("*** Error loading weights ***");
		return 1;
	}

	char f_name[F_NAME_SIZE];
	FILE *f_index = fopen("/export/space/test_val_set/goldendata/inputs/index.txt","r");
	FILE *f_results = fopen("/export/space/test_val_set/goldendata/inputs/labels.txt","r");
//	FILE *f_index = fopen("/export/space/test_val_set/goldendata/inputs/index_1.txt","r");
//	FILE *f_results = fopen("/export/space/test_val_set/goldendata/inputs/labels_1.txt","r");
	FILE *f_input;


	if (f_index == NULL) {
		printf("*** Error getting index file, exiting\n");
		return 1;
	}
	if (f_results == NULL) {
		printf("*** Error getting results file, exiting\n");
		return 1;
	}

	while (fgets(f_name,F_NAME_SIZE,f_index) != NULL) {
		rm_nl(f_name);

		f_input = fopen(f_name,"r");
		if (f_input == NULL) {
			printf("*** Error, file %s couldn't be opened\n",f_name);
			continue;
		}

		golden_result = read_in_data_and_golden_res(f_input,input,f_results);

		hw_result = fpga_top(input,false,bias,kernels);

		accuracy = get_output_statistics(golden_result,hw_result,f_name);

		fclose(f_input);

	}

	if (accuracy < MIN_ACCURACY)
		errorn = 1;
	else
		errorn = 0;

	printf("*** Final accuracy is %f ***\n",accuracy*100);
	fclose(f_index);
	fclose(f_results);
    return errorn;
}
#endif
