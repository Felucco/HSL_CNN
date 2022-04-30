#ifndef __FLATTEN
#define __FLATTEN

#include "activations.h"

template<uint32_t IN_SHAPE_H, uint32_t IN_SHAPE_W, uint32_t IN_SHAPE_CH>
void flatten(float out [IN_SHAPE_H*IN_SHAPE_W*IN_SHAPE_CH], float in [IN_SHAPE_H][IN_SHAPE_W][IN_SHAPE_CH])
{
    uint32_t out_idx = 0;
    uint16_t in_row, in_col, in_ch;

    flat_row_for: for (in_row=0; in_row < IN_SHAPE_H; in_row++)
    {
        flat_col_for: for (in_col=0; in_col < IN_SHAPE_W; in_col++)
        {
            flat_ch_for: for (in_ch=0; in_ch < IN_SHAPE_CH; in_ch++)
            {
                out[out_idx] = in[in_row][in_col][in_ch];
                out_idx++;
            }
        }
    }
}

#endif