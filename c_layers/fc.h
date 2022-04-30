#include "defines.h"
#include "activations.h"

template<uint32_t IN_LEN, uint16_t NEURONS>
void fc(float24_t out [NEURONS], float24_t in [IN_LEN], float24_t weight[IN_LEN][NEURONS], float24_t bias[NEURONS]);

template<uint32_t IN_LEN, uint16_t NEURONS>
void fc_norelu(float24_t out [NEURONS], float24_t in [IN_LEN], float24_t weight[IN_LEN][NEURONS], float24_t bias[NEURONS]);