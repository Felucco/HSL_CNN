#ifndef __CONV
#define __CONV

#include "defines.h"
#include "activations.h"

//Stride supposed 1 for now

template<uint32_t IN_SHAPE_H, uint32_t IN_SHAPE_W, uint32_t IN_SHAPE_CH, uint8_t KERNEL_SIZE, uint8_t FILTERS>
void conv(float out[IN_SHAPE_H-KERNEL_SIZE+1][IN_SHAPE_W-KERNEL_SIZE+1][FILTERS], float in[IN_SHAPE_H][IN_SHAPE_W][IN_SHAPE_CH], float weight[KERNEL_SIZE][KERNEL_SIZE][IN_SHAPE_CH][FILTERS], float bias[FILTERS])
{
	//constexpr int BUFFER_SIZE = IN_SHAPE_W * IN_SHAPE_CH * (KERNEL_SIZE -1) + KERNEL_SIZE * IN_SHAPE_CH

	uint16_t filt,out_row,out_col,k_row,k_col, ch;
	float sum, in_val;
	//int row_offset, col_offset, channel_offset;

	conv_filter_for:for(filt=0; filt<FILTERS; filt++)
	{
		conv_row_for:for(out_row = 0; out_row < (IN_SHAPE_H-KERNEL_SIZE+1); out_row += 1)
		{
			conv_col_for:for(out_col = 0; out_col < (IN_SHAPE_W-KERNEL_SIZE+1); out_col += 1)
			{

				sum = 0;

				conv_channel_for:for(ch=0; ch<IN_SHAPE_CH; ch += 1)
				{
					conv_ker_col_for:for(k_col = 0; k_col < KERNEL_SIZE; k_col ++)
					{
						conv_ker_row_for:for(k_row = 0; k_row < KERNEL_SIZE; k_row ++)
						{
							in_val = in[out_row+k_row][out_col+k_col][ch];
							sum += in_val * weight[k_row][k_col][ch][filt];
						}
					}
				}
				out[out_row][out_col][filt]=relu(sum+bias[filt]);
			}
		}
	}
}

#endif