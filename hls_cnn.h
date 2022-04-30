#include "headers/defines.h"
#include "headers/activations.h"
#include "headers/weights.h"

#include "c_layers/conv.h"
#include "c_layers/fc.h"
#include "c_layers/flatten.h"
#include "c_layers/pool.h"

void hls_cnn(float in_image [IMAGE_SIZE][IMAGE_SIZE][IMAGE_CHANNELS], float out_tensor [FC3_ACT_SIZE]);