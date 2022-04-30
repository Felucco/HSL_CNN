/******************************************************************************
 * (C) Copyright 2020 AMIQ Consulting
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * NAME:        conv.cpp
 * PROJECT:     conv
 * Description: HLS implementation of the Convolutional Layer
 *******************************************************************************/

#include "conv.h"

#ifndef __SYNTHESIS__
#include <stdio.h>
#endif

template<uint32_t IN_SHAPE_H, uint32_t IN_SHAPE_W, uint32_t IN_SHAPE_CH, uint8_t KERNEL_SIZE, uint8_t FILTERS>
void conv(float out[IN_SHAPE_H-KERNEL_SIZE+1][IN_SHAPE_W-KERNEL_SIZE+1][FILTERS], float in[IN_SHAPE_H][IN_SHAPE_W][IN_SHAPE_CH], float weight[KERNEL_SIZE][KERNEL_SIZE][IN_SHAPE_CH][FILTERS], float bias[FILTERS])
{
	//constexpr int BUFFER_SIZE = IN_SHAPE_W * IN_SHAPE_CH * (KERNEL_SIZE -1) + KERNEL_SIZE * IN_SHAPE_CH

	constexpr unsigned MIN_Y = (KERNEL_SIZE-1)/2;
	constexpr unsigned MAX_Y = IN_SHAPE_H-1-MIN_Y;
	constexpr unsigned MIN_X = MIN_Y;
	constexpr unsigned MAX_X = IN_SHAPE_W-1-MIN_W;


	uint16_t filt,out_row,out_col,k_row,k_col, ch;
	float sum;
	//int row_offset, col_offset, channel_offset;

	conv_filter_for:for(filt=0; filt<FILTERS; filt++)
	{
		conv_row_for:for(out_row = 0; out_row < (IN_SHAPE_H-KERNEL_SIZE+1); out_row += 1)
		{
			conv_col_for:for(out_col = 0; out_row < (IN_SHAPE_W-KERNEL_SIZE+1); out_col += 1)
			{

				sum = 0;

				conv_channel_for:for(ch=0; ch<IN_SHAPE_CH; ch += 1)
				{
					conv_ker_col_for:for(k_col = 0; k_col < KERNEL_SIZE; k_col ++)
					{
						conv_ker_row_for:for(k_row = 0; k_row < KERNEL_SIZE; k_row ++)
						{
							sum += in[out_col+k_col][out_row+k_row][ch] * weight[k_col][k_row][ch][filt];
						}
					}
				}
				out[out_row][out_col][filt]=relu(sum+bias[filt]);
			}
		}
	}
}
