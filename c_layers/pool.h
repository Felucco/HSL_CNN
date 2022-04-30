
#include "defines.h"
#include "activations.h"

//Simple MaxPool: stride = kernel size

template<uint32_t IN_SHAPE_H, uint32_t IN_SHAPE_W, uint32_t IN_SHAPE_CH, uint8_t KERNEL_SIZE>
void pool(float24_t out [IN_SHAPE_H/KERNEL_SIZE][IN_SHAPE_W/KERNEL_SIZE][IN_SHAPE_CH], float24_t in[IN_SHAPE_H][IN_SHAPE_W][IN_SHAPE_CH]);