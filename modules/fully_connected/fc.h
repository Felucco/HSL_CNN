#include "headers/defines.h"
#include "headers/activations.h"
#include <hls_stream.h>

void fc(hls::stream<float> &out, hls::stream<float> &in, float weight[FC_WEIGHTS_H][FC_WEIGHTS_W], float bias[FC_BIAS_SIZE]);