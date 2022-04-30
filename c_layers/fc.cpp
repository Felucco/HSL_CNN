/******************************************************************************
 * (C) Copyright 2020 AMIQ Consulting
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * NAME:        fc.cpp
 * PROJECT:     fully_connected
 * Description: HLS implementation of the Fully Connected Layer
 *******************************************************************************/

#include "fc.h"

/*
	Legge valore per valore i float in ingresso provenienti dal layer precedente. Per ogni valore (in numero pari alla larghezza post-flatten del layer precedente)
	(for esterno) viene mappato l'impatto su tutti i neuroni del layer dense secondo i rispettivi pesi (for interno).
	Alla fine viene aggiunto ad ogni neurone il suo bias e viene computata l'attivazione
*/
template<uint32_t IN_LEN, uint16_t NEURONS>
void fc(float24_t out [NEURONS], float24_t in [IN_LEN], float24_t weight[IN_LEN][NEURONS], float24_t bias[NEURONS])
{
	float24_t sum, in_pos, neur;

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
void fc_norelu(float24_t out [NEURONS], float24_t in [IN_LEN], float24_t weight[IN_LEN][NEURONS], float24_t bias[NEURONS])
{
	float24_t sum, in_pos, neur;

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