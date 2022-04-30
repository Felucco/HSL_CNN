#include <stdio.h>

#define TEST_N 10
#define TB_IMAGE_SIZE 			32
#define TB_IMAGE_CHANNELS		1
#define OUT_NEURONS             10

extern void hls_cnn(float in_image [TB_IMAGE_SIZE][TB_IMAGE_SIZE][TB_IMAGE_CHANNELS], float out_tensor [OUT_NEURONS]);

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

    float in_imgs [TEST_N][TB_IMAGE_SIZE][TB_IMAGE_SIZE][TB_IMAGE_CHANNELS];
    float out_tensors [TEST_N][OUT_NEURONS];
    float exp_tensors [TEST_N][OUT_NEURONS];

    //Load input images
    FILE *img_file = fopen("refs/inputs_ref.dat","r");
    for (test = 0; test < TEST_N; test++)
    {
        for (row=0; row < TB_IMAGE_SIZE; row++)
        {
            for (col=0; col < TB_IMAGE_SIZE; col++)
            {
                for (ch = 0; ch < TB_IMAGE_CHANNELS; ch++){
                    fscanf(img_file, "%f ",&(in_imgs[test][row][col][ch]));
                }
            }
        }
        fscanf(img_file, "\n", NULL);
    }
    fclose(img_file);

    //Load expected outputs
    //printf("\n-------------Expected outputs:-------------\n");
    FILE *exp_file = fopen("refs/fc_layer3_ref.dat","r");
    for (test = 0; test < TEST_N; test++)
    {
        for (row = 0; row < OUT_NEURONS; row++){
            fscanf(exp_file,"%f ", &(exp_tensors[test][row]));
        }
        fscanf(exp_file, "\n", NULL);
        //print_tensor(exp_tensors[test],OUT_NEURONS);
        //printf("\n");
    }
    fclose(exp_file);

    for (test = 0; test < TEST_N; test++){
        hls_cnn(in_imgs[test],out_tensors[test]);
        printf("------------Test %d--------------\nExpected: ",test);
        print_tensor(exp_tensors[test],OUT_NEURONS);
        printf("\nReal: ");
        print_tensor(out_tensors[test],OUT_NEURONS);
        printf("\n\n");
    }  
}
