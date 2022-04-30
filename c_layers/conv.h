#include "defines.h"
#include "activations.h"
#include "buffer.h"

//Stride supposed 1 for now

template<uint32_t IN_SHAPE_H, uint32_t IN_SHAPE_W, uint32_t IN_SHAPE_CH, uint8_t KERNEL_SIZE, uint8_t FILTERS>
void conv(float24_t out[IN_SHAPE_H-KERNEL_SIZE+1][IN_SHAPE_W-KERNEL_SIZE+1][FILTERS], float24_t in[IN_SHAPE_H][IN_SHAPE_W][IN_SHAPE_CH], float24_t weight[KERNEL_SIZE][KERNEL_SIZE][IN_SHAPE_CH][FILTERS], float24_t bias[FILTERS]);