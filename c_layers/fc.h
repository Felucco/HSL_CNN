#ifndef __FC
#define __FC

#include "defines.h"
#include "activations.h"

template<uint32_t IN_LEN, uint16_t NEURONS>
void fc(float out [NEURONS], float in [IN_LEN], float weight[IN_LEN][NEURONS], float bias[NEURONS])
{
	float sum;
	uint32_t in_pos;
	uint16_t neur;

	fc_neur_for:for (neur = 0; neur < NEURONS; neur++)
	{
		sum = 0;
		fc_input_for:for (in_pos = 0; in_pos<IN_LEN; in_pos++)
		{
			sum += in[in_pos]*weight[in_pos][neur];
		}
		out[neur]=relu(sum+bias[neur]);
	}
}

template<uint32_t IN_LEN, uint16_t NEURONS>
void fc_norelu(float out [NEURONS], float in [IN_LEN], float weight[IN_LEN][NEURONS], float bias[NEURONS])
{
	float sum;
	uint32_t in_pos;
	uint16_t neur;

	fc_neur_for:for (neur = 0; neur < NEURONS; neur++)
	{
		sum = 0;
		fc_input_for:for (in_pos = 0; in_pos<IN_LEN; in_pos++)
		{
			sum += in[in_pos]*weight[in_pos][neur];
		}
		out[neur]=sum+bias[neur];
	}
}

#endif