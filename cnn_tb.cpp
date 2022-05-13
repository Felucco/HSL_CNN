#include <stdio.h>
#include <math.h>

#define TEST_N 10
#define TB_IMAGE_SIZE 			32
#define TB_IMAGE_CHANNELS		1
#define OUT_NEURONS             10

#define MAX_ERROR               0.0001

extern void hls_cnn(float in_image [TB_IMAGE_SIZE][TB_IMAGE_SIZE][TB_IMAGE_CHANNELS], float out_tensor [OUT_NEURONS]);

void print_tensor(float *tensor, int len){
    printf("[");
    for (size_t i = 0; i < len-1; i++)
    {
        printf("%.5f ", (float)tensor[i]);
    }
    printf("%.5f]", (float)tensor[len-1]);    
}

int check_tensor(float exp [OUT_NEURONS], float real [OUT_NEURONS]){
    int errors = 0;
    float err;
    for (size_t i = 0; i < OUT_NEURONS; i++)
    {
        err = fabs(exp[i] - real[i]);
        if (err > MAX_ERROR) errors++;
    }
    return errors;
}

int main()
{
    printf("\n---------------------------Initializing Test Bench---------------------------\n");
    unsigned test,row,col,ch;

    float in_imgs [TEST_N][TB_IMAGE_SIZE][TB_IMAGE_SIZE][TB_IMAGE_CHANNELS];
    float out_tensors [TEST_N][OUT_NEURONS];
    float exp_tensors [TEST_N][OUT_NEURONS];

    //Load input images
    FILE *img_file = fopen("inputs_ref.dat","r");
    if (img_file == NULL){
        printf("Could open input image file, aborting...");
        return -1;
    }
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
    printf("Input images loaded\n");

    //Load expected outputs
    //printf("\n-------------Expected outputs:-------------\n");
    FILE *exp_file = fopen("fc_layer3_ref.dat","r");
    if (exp_file == NULL){
        printf("Could open expected tensors file, aborting...");
        return -1;
    }
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

    printf("Expected outputs loaded\n");
    printf("\n---------------------------Starting evaluation---------------------------\n\n");

    int err, total_errors = 0;

    for (test = 0; test < TEST_N; test++){
        hls_cnn(in_imgs[test],out_tensors[test]);
        err = check_tensor(exp_tensors[test],out_tensors[test]);
        total_errors += err;

        printf("---------------------------Test %d---------------------------\nExpected: ",test+1);
        print_tensor(exp_tensors[test],OUT_NEURONS);
        printf("\nReal: ");
        print_tensor(out_tensors[test],OUT_NEURONS);
        printf("\nChecking with MAX ERROR = %f --> %d errors detected\n\n",MAX_ERROR,err);
    }

    printf("\n----------------TB COMPLETED----------------\n\n");
    if (total_errors > 0) printf("%d errors detected with MAX ERROR = %f", total_errors, MAX_ERROR);
    else printf("Every entry was found compliant with MAX ERROR = %f", MAX_ERROR);
    printf("\n\n----------------------------------------------------\n\n");

    return total_errors;
}
