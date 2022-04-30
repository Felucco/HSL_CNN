#include "activations.h"

template<uint32_t IN_SHAPE_H, uint32_t IN_SHAPE_W, uint32_t IN_SHAPE_CH>
void flatten(float24_t out [IN_SHAPE_H*IN_SHAPE_W*IN_SHAPE_CH], float24_t in [IN_SHAPE_H][IN_SHAPE_W][IN_SHAPE_CH]);