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

#define CONV_1x1 0
#define CONV_3x3 1


#define CONV1  {1,CONV_3x3,2,3,64,256,128,0,-2,7}
#define FIRE2_SQUEEZE3X3  {2,CONV_3x3,2,64,16,128,64,-2,-4,6}
#define FIRE2_EXPAND1X1  {3,CONV_1x1,1,16,64,64,64,-4,-4,6}
#define FIRE2_EXPAND3X3  {4,CONV_3x3,1,16,64,64,64,-4,-4,7}
#define FIRE3_SQUEEZE1X1  {5,CONV_1x1,1,128,16,64,64,-4,-6,6}
#define FIRE3_EXPAND1X1  {6,CONV_1x1,1,16,64,64,64,-6,-5,7}
#define FIRE3_EXPAND3X3  {7,CONV_3x3,1,16,64,64,64,-6,-5,7}
#define FIRE4_SQUEEZE3X3  {8,CONV_3x3,2,128,32,64,32,-5,-7,8}
#define FIRE4_EXPAND1X1  {9,CONV_1x1,1,32,128,32,32,-7,-6,7}
#define FIRE4_EXPAND3X3  {10,CONV_3x3,1,32,128,32,32,-7,-6,8}
#define FIRE5_SQUEEZE1X1  {11,CONV_1x1,1,256,32,32,32,-6,-7,7}
#define FIRE5_EXPAND1X1  {12,CONV_1x1,1,32,128,32,32,-7,-6,7}
#define FIRE5_EXPAND3X3  {13,CONV_3x3,1,32,128,32,32,-7,-6,7}
#define FIRE6_SQUEEZE3X3  {14,CONV_3x3,2,256,64,32,16,-6,-7,8}
#define FIRE6_EXPAND1X1  {15,CONV_1x1,1,64,256,16,16,-7,-6,7}
#define FIRE6_EXPAND3X3  {16,CONV_3x3,1,64,256,16,16,-7,-6,8}
#define FIRE7_SQUEEZE1X1  {17,CONV_1x1,1,512,64,16,16,-6,-7,7}
#define FIRE7_EXPAND1X1  {18,CONV_1x1,1,64,192,16,16,-7,-5,8}
#define FIRE7_EXPAND3X3  {19,CONV_3x3,1,64,192,16,16,-7,-5,8}
#define FIRE8_SQUEEZE3X3  {20,CONV_3x3,2,384,112,16,8,-5,-6,8}
#define FIRE8_EXPAND1X1  {21,CONV_1x1,1,112,256,8,8,-6,-4,8}
#define FIRE8_EXPAND3X3  {22,CONV_3x3,1,112,256,8,8,-6,-4,8}
#define FIRE9_SQUEEZE1X1  {23,CONV_1x1,1,512,112,8,8,-4,-4,8}
#define FIRE9_EXPAND1X1  {24,CONV_1x1,1,112,368,8,8,-4,-2,8}
#define FIRE9_EXPAND3X3  {25,CONV_3x3,1,112,368,8,8,-4,-2,9}
#define CONV10_TRANSFER  {26,CONV_1x1,1,736,26,8,8,-2,-1,9}
#define NET {CONV1,FIRE2_SQUEEZE3X3,FIRE2_EXPAND1X1,FIRE2_EXPAND3X3,FIRE3_SQUEEZE1X1,FIRE3_EXPAND1X1,FIRE3_EXPAND3X3,FIRE4_SQUEEZE3X3,FIRE4_EXPAND1X1,FIRE4_EXPAND3X3,FIRE5_SQUEEZE1X1,FIRE5_EXPAND1X1,FIRE5_EXPAND3X3,FIRE6_SQUEEZE3X3,FIRE6_EXPAND1X1,FIRE6_EXPAND3X3,FIRE7_SQUEEZE1X1,FIRE7_EXPAND1X1,FIRE7_EXPAND3X3,FIRE8_SQUEEZE3X3,FIRE8_EXPAND1X1,FIRE8_EXPAND3X3,FIRE9_SQUEEZE1X1,FIRE9_EXPAND1X1,FIRE9_EXPAND3X3,CONV10_TRANSFER,}

#define IM_PX_SZ 256
#define IM_CH_IN 3
#define OUT_PX 8
#define OUT_CH 26

#define MAX_OUTPUT_SIZE 1048576
#define MAX_INPUT_SIZE 196608
#define TOTAL_BIAS 3450
#define TOTAL_WEIGHTS 198876
#define MAX_CH_OUT 736

#define TOTAL_OPS 26
