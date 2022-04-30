#include "defines.h"
#include "activations.h"

template<uint32_t IN_LEN, uint16_t NEURONS>
void fc(float out [NEURONS], float in [IN_LEN], float weight[IN_LEN][NEURONS], float bias[NEURONS]);

template<uint32_t IN_LEN, uint16_t NEURONS>
void fc_norelu(float out [NEURONS], float in [IN_LEN], float weight[IN_LEN][NEURONS], float bias[NEURONS]);