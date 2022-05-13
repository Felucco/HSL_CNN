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
 * NAME:        defines.h
 * PROJECT:     nnet_stream
 * Description: CNN related parameters (hyper-parameters)
 *******************************************************************************/


#ifndef __DEFINES_H
#define __DEFINES_H

#define IMAGE_SIZE 			32
#define IMAGE_CHANNELS		1

//First layer: 2D convolutional layer, 5x5, 8 filters, stride=1, padding valid. ReLu activation
//Output shape: 28x28x8

#define CONV1_KERNEL_SIZE 	5
#define CONV1_CHANNELS 		1
#define CONV1_FILTERS 		8
#define CONV1_BIAS_SIZE 	8
#define CONV1_STRIDE 		1

#define A1_SIZE				28
#define A1_CHANNELS			8

//Second layer: 2D MaxPool layer, 2x2
//Output shape: 14x14x8

#define P1_SIZE				14
#define P1_CHANNELS			8
#define P1_KERNEL_SIZE		2
#define P1_STRIDE			2

//Third layer: 2D convolutional layer, 3x3, 16 filters, stride=1, padding valid. ReLu activation
//Output shape: 12x12x16

#define CONV2_KERNEL_SIZE 	3
#define CONV2_CHANNELS 		8
#define CONV2_FILTERS 		16
#define CONV2_BIAS_SIZE		16
#define CONV2_STRIDE 		1

#define A2_SIZE				12
#define A2_CHANNELS			16

//Fourth layer: 2D MaxPool layer, 2x2
//Output shape: 6x6x16

#define P2_SIZE				6
#define P2_CHANNELS			16
#define P2_KERNEL_SIZE		2
#define P2_STRIDE			2

//Fifth layer: flatten. Output shape 6x6x16=576 flat neurons

#define FLATTEN_SIZE		576

//Sixth layer: Fully Connected / Dense layer, 120 neurons, ReLu

#define FC1_WEIGHTS_H		576
#define FC1_WEIGHTS_W		120
#define FC1_BIAS_SIZE		120

#define FC1_ACT_SIZE		120

//Seventh layer: Fully Connected / Dense layer, 84 neurons, ReLu

#define FC2_WEIGHTS_H		120
#define FC2_WEIGHTS_W		84
#define FC2_BIAS_SIZE		84

#define FC2_ACT_SIZE		84

//Last layer: Fully Connected / Dense layer, 10 neurons, Linear activation (no activation)

#define FC3_WEIGHTS_H		84
#define FC3_WEIGHTS_W		10
#define FC3_BIAS_SIZE		10

#define FC3_ACT_SIZE		10



#endif