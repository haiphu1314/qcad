/**
 * @ Author: Hai Phu
 * @ Email:  haiphu@hcmut.edu.vn
 * @ Create Time: 2024-06-27 16:23:39
 * @ Modified time: 2024-09-19 17:40:23
 * @ Description:
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "src/model.h"
// #include "testcase.h"
#include <time.h>
#include <nmmintrin.h>
#define NUM_TESTCASES 84000
// #define NO_TESTS 1
#define NO_TESTS 1
#define NUM_CPU 24

int power(int base, int exponent)
{
    int result = 1;
    for (int i = 0; i < exponent; i++)
    {
        result *= base;
    }
    return result;
}
float getRandomNumber()
{
    return (float)rand() / RAND_MAX * 2.0f - 1.0f;
}
int main()
{
    printf("ALEXXXXXXXXXXXXXXX\n");
    // omp_set_num_threads(NUM_CPU);
    quant_type typ[4] = {FP, TNN, TBN, BNN};
    int len_typ = 4;
    double time_FP;
    for (int t = 0; t < len_typ; t++)
    // int t = 1;
    {
        int input_height = 224;
        int input_width = 224;
        int input_channel = 3;
        layer_node *model = NULL;
        conv2d_layer *conv1 = create_conv2d_layer(3, 64, 11, 4, 2, 1, typ[t]);
        model = add_layer(model, CONV, "conv1", conv1);
        conv2d_layer *conv2 = create_conv2d_layer(64, 192, 5, 1, 2, 1, typ[t]);
        model = add_layer(model, CONV, "conv2", conv2);
        conv2d_layer *conv3 = create_conv2d_layer(128, 384, 3, 1, 1, 1, typ[t]);
        model = add_layer(model, CONV, "conv3", conv3);

        conv2d_layer *conv4 = create_conv2d_layer(384, 256, 3, 1, 1, 1, typ[t]);
        model = add_layer(model, CONV, "conv4", conv4);        
        conv2d_layer *conv5 = create_conv2d_layer(256, 256, 3, 1, 1, 1, typ[t]);
        model = add_layer(model, CONV, "conv5", conv5);

        linear_layer *linear1 = create_linear_layer(256 * 6 * 6, 4096, typ[t]);
        model = add_layer(model, LINEAR, "linear1", linear1);
        linear_layer *linear2 = create_linear_layer(4096, 4096, typ[t]);
        model = add_layer(model, LINEAR, "linear2", linear2);
        linear_layer *linear3 = create_linear_layer(4096, 1000, typ[t]);
        model = add_layer(model, LINEAR, "linear3", linear3);
        srand(time(NULL));
        struct timeval start, end;
        gettimeofday(&start, NULL);

        for (int ct = 0; ct < NO_TESTS; ct++)
        {
            float *input = (float*)malloc(input_channel * input_height * input_width * sizeof(float*));
            for (int c = 0; c < 1; c++)
            {
                for (int h = 0; h < input_height; h++)
                {
                    for (int w = 0; w < input_width; w++)
                    {
                        input[c*h*w] = power(-1, w);
                    }
                }
            }
            float *x1 = conv2d_forward(conv1, input, input_height, input_width);
            float *x1_mp = max_pooling_2d_k(x1, conv1->output_channel, 55, 55, 3, 2);
            float *x2 = conv2d_forward(conv2, x1_mp, 27, 27);
            float *x2_mp = max_pooling_2d_k(x2, conv2->output_channel, 27, 27, 3, 2);

            float *x3 = conv2d_forward(conv3, x2_mp, 13, 13);
            float *x4 = conv2d_forward(conv4, x3, 13, 13);
            float *x5 = conv2d_forward(conv5, x4, 13, 13);
            float *x5_mp = max_pooling_2d_k(x5, conv5->output_channel, 13, 13, 3, 2);

            float *input_linear = flatto1d(x5_mp, 256, 6, 6);
            float *x6 = linear_forward(linear1, input_linear);
            float *x7 = linear_forward(linear2, x6);
            float *x8 = linear_forward(linear2, x7);
        }
        gettimeofday(&end, NULL);
        double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
        if(typ[t] == FP){
            time_FP = time_taken;
        }
        printf("Thời gian thực thi mô hình %d: %.6f giây, %.4f\n\n", t, time_taken/NO_TESTS, time_FP/time_taken);
    }

}
