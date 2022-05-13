#include "hls_cnn.h"

void hls_cnn(float in_image [IMAGE_SIZE][IMAGE_SIZE][IMAGE_CHANNELS], float out_tensor [FC3_ACT_SIZE])
{
    float conv1_out [A1_SIZE][A1_SIZE][A1_CHANNELS];
    float pool1_out [P1_SIZE][P1_SIZE][P1_CHANNELS];
    float conv2_out [A2_SIZE][A2_SIZE][A2_CHANNELS];
    float pool2_out [P2_SIZE][P2_SIZE][P2_CHANNELS];
    float flatten_out [FLATTEN_SIZE];
    float fc1_out [FC1_ACT_SIZE];
    float fc2_out [FC2_ACT_SIZE];


    conv<IMAGE_SIZE,IMAGE_SIZE,IMAGE_CHANNELS,CONV1_KERNEL_SIZE,CONV1_FILTERS>(conv1_out,in_image,conv_layer1_weights,conv_layer1_bias);
    pool<A1_SIZE,A1_SIZE,A1_CHANNELS,P1_KERNEL_SIZE>(pool1_out,conv1_out);

    conv2<P1_SIZE,P1_SIZE,P1_CHANNELS,CONV2_KERNEL_SIZE,CONV2_FILTERS>(conv2_out,pool1_out,conv_layer2_weights,conv_layer2_bias);
    pool<A2_SIZE,A2_SIZE,A2_CHANNELS,P2_KERNEL_SIZE>(pool2_out,conv2_out);

    flatten<P2_SIZE,P2_SIZE,P2_CHANNELS>(flatten_out,pool2_out);

    fc<FLATTEN_SIZE,FC1_ACT_SIZE>(fc1_out,flatten_out,fc_layer1_weights,fc_layer1_bias);
    fc<FC1_ACT_SIZE,FC2_ACT_SIZE>(fc2_out,fc1_out,fc_layer2_weights,fc_layer2_bias);
    fc_norelu<FC2_ACT_SIZE,FC3_ACT_SIZE>(out_tensor,fc2_out,fc_layer3_weights,fc_layer3_bias);

}
