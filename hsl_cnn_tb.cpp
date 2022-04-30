#include "hls_cnn.h"

#include <stdio.h>

#define TEST_N 10

void print_tensor(float *tensor, int len){
    printf("[");
    for (size_t i = 0; i < len-1; i++)
    {
        printf("%.3f ", (float)tensor[i]);
    }
    printf("%.3f]", (float)tensor[len-1]);    
}

int main()
{
    unsigned test,row,col,ch;

    float in_imgs [TEST_N][IMAGE_SIZE][IMAGE_SIZE][IMAGE_CHANNELS];
    float out_tensors [TEST_N][FC3_ACT_SIZE];
    float exp_tensors [TEST_N][FC3_ACT_SIZE];

    //Load input images
    FILE *img_file = fopen("inputs_ref.dat","r");
    for (test = 0; test < TEST_N; test++)
    {
        for (row=0; row < IMAGE_SIZE; row++)
        {
            for (col=0; col < IMAGE_SIZE; col++)
            {
                for (ch = 0; ch < IMAGE_CHANNELS; ch++){
                    fscanf(img_file, "%f ",&(in_imgs[test][row][col][ch]));
                }
            }
        }
        fscanf(img_file, "\n", NULL);
    }
    fclose(img_file);

    //Load expected outputs
    FILE *exp_file = fopen("fc_layer3_ref.dat","r");
    for (test = 0; test < TEST_N; test++)
    {
        for (row = 0; row < FC3_ACT_SIZE; row++){
            fscanf(exp_file,"%f ", &(exp_tensors[test][row]));
        }
        fscanf(exp_file, "\n", NULL);
    }
    fclose(exp_file);

    for (test = 0; test < TEST_N; test++){
        hls_cnn(in_imgs[test],out_tensors[test]);
        printf("------------Test %d--------------\nExpected: ");
        print_tensor(exp_tensors[test],FC3_ACT_SIZE);
        printf("\nReal: ");
        print_tensor(out_tensors[test],FC3_ACT_SIZE);
        printf("\n\n");
    }  
}
