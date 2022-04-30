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
 * NAME:        pool.cpp
 * PROJECT:     pool
 * Description: HLS implementation of the MaxPooling Layer
 *******************************************************************************/

#include "pool.h"

template<uint32_t IN_SHAPE_H, uint32_t IN_SHAPE_W, uint32_t IN_SHAPE_CH, uint8_t KERNEL_SIZE>
void pool(float out [IN_SHAPE_H/KERNEL_SIZE][IN_SHAPE_W/KERNEL_SIZE][IN_SHAPE_CH], float in[IN_SHAPE_H][IN_SHAPE_W][IN_SHAPE_CH])
{

	uint16_t out_row,out_col,k_row,k_col,ch;
	float max_val, tmp_val;

	pool_ch_for: for (ch = 0; ch < IN_SHAPE_CH; ch++)
	{
		pool_row_for: for (out_row = 0; out_row < (IN_SHAPE_H/KERNEL_SIZE); out_row++)
		{
			pool_col_for: for (out_col = 0; out_col < (IN_SHAPE_W/KERNEL_SIZE); out_col++)
			{
				max_val = in[out_row*KERNEL_SIZE][out_col*KERNEL_SIZE][ch];
				pool_ker_row_for: for (k_row = 0; k_row < KERNEL_SIZE; k_row++)
				{
					pool_ker_col_for: for (k_col = 0; k_col < KERNEL_SIZE; k_col++)
					{
						tmp_val = in[out_row*KERNEL_SIZE+k_row][out_col*KERNEL_SIZE+k_col][ch];
						max_val = tmp_val > max_val ? tmp_val : max_val;
					}
				}
				out[out_row][out_col][ch] = max_val;
			}
		}
	}

}


