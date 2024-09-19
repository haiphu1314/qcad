/**
 * @ Author: Hai Phu
 * @ Email:  haiphu@hcmut.edu.vn
 * @ Create Time: 2024-06-27 16:23:39
 * @ Modified time: 2024-08-01 23:59:44
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
    quant_type typ[4] = {BNN, TNN, TBN, BNN};
    int len_typ = 4;
    for (int t = 0; t < len_typ; t++)
    // int t = 1;
    {
        int input_height = 64;
        int input_width = 64;
        int input_channels = 3;
        layer_node *model = NULL;
        conv2d_layer *conv1_1 = create_conv2d_layer(input_channels, 64, 3, 1, 1, 1, typ[t]);
        model = add_layer(model, CONV, "conv1_1", conv1_1);
        conv2d_layer *conv1_2 = create_conv2d_layer(64, 64, 3, 1, 1, 1, typ[t]);
        model = add_layer(model, CONV, "conv1_2", conv1_2);

        conv2d_layer *conv2_1 = create_conv2d_layer(64, 128, 3, 1, 1, 1, typ[t]);
        model = add_layer(model, CONV, "conv2_1", conv2_1);
        conv2d_layer *conv2_2 = create_conv2d_layer(128, 128, 3, 1, 1, 1, typ[t]);
        model = add_layer(model, CONV, "conv2_2", conv2_2);

        conv2d_layer *conv3_1 = create_conv2d_layer(128, 256, 3, 1, 1, 1, typ[t]);
        model = add_layer(model, CONV, "conv3_1", conv3_1);
        conv2d_layer *conv3_2 = create_conv2d_layer(256, 256, 3, 1, 1, 1, typ[t]);
        model = add_layer(model, CONV, "conv3_2", conv3_2);
        conv2d_layer *conv3_3 = create_conv2d_layer(256, 256, 3, 1, 1, 1, typ[t]);
        model = add_layer(model, CONV, "conv3_3", conv3_3);

        conv2d_layer *conv4_1 = create_conv2d_layer(256, 512, 3, 1, 1, 1, typ[t]);
        model = add_layer(model, CONV, "conv4_1", conv4_1);
        conv2d_layer *conv4_2 = create_conv2d_layer(512, 512, 3, 1, 1, 1, typ[t]);
        model = add_layer(model, CONV, "conv4_2", conv4_2);
        conv2d_layer *conv4_3 = create_conv2d_layer(512, 512, 3, 1, 1, 1, typ[t]);
        model = add_layer(model, CONV, "conv4_3", conv4_3);

        conv2d_layer *conv5_1 = create_conv2d_layer(512, 512, 3, 1, 1, 1, typ[t]);
        model = add_layer(model, CONV, "conv5_1", conv5_1);
        conv2d_layer *conv5_2 = create_conv2d_layer(512, 512, 3, 1, 1, 1, typ[t]);
        model = add_layer(model, CONV, "conv5_2", conv5_2);
        conv2d_layer *conv5_3 = create_conv2d_layer(512, 512, 3, 1, 1, 1, typ[t]);
        model = add_layer(model, CONV, "conv5_3", conv5_3);
        
        linear_layer *fc1 = create_linear_layer(512 * 7 * 7, 4096, typ[t]);
        model = add_layer(model, LINEAR, "fc1", fc1);
        linear_layer *fc2 = create_linear_layer(4096, 4096, typ[t]);
        model = add_layer(model, LINEAR, "fc2", fc2);
        linear_layer *fc3 = create_linear_layer(4096, 10, typ[t]);
        model = add_layer(model, LINEAR, "fc3", fc3);

        srand(time(NULL));
        clock_t start, end;
        start = clock();

        for (int ct = 0; ct < NO_TESTS; ct++)
        {
            float ***input = allocate_3d_float_array(input_channels, input_height, input_width);
            for (int c = 0; c < 1; c++)
            {
                for (int h = 0; h < input_height; h++)
                {
                    for (int w = 0; w < input_width; w++)
                    {
                        input[c][h][w] = power(-1, w);
                    }
                }
            }
            int x1_height = input_height;
            int x1_width = input_width;

            float ***x1_1 = conv2d_forward(conv1_1, input, x1_height, x1_width);
            float ***x1_2 = conv2d_forward(conv1_2, x1_1, x1_height, x1_width);
            float ***x1_mp = max_pooling_2d(x1_2, conv1_2->output_channel, x1_height, x1_height);

            int x2_height = x1_height/2;
            int x2_width = x1_width/2;
            float ***x2_1 = conv2d_forward(conv2_1, x1_mp, x2_height, x2_width);
            float ***x2_2 = conv2d_forward(conv2_2, x2_1, x2_height, x2_width);
            float ***x2_mp = max_pooling_2d(x2_2, conv2_2->output_channel, x2_height, x2_height);

            int x3_height = x2_height/2;
            int x3_width = x2_width/2;
            float ***x3_1 = conv2d_forward(conv3_1, x2_mp, x3_height, x3_width);
            float ***x3_2 = conv2d_forward(conv3_2, x3_1, x3_height, x3_width);
            float ***x3_3 = conv2d_forward(conv3_3, x3_2, x3_height, x3_width);
            float ***x3_mp = max_pooling_2d(x3_3, conv3_2->output_channel, x3_height, x3_height);

            int x4_height = x3_height/2;
            int x4_width = x3_width/2;
            float ***x4_1 = conv2d_forward(conv4_1, x3_mp, x4_height, x4_width);
            float ***x4_2 = conv2d_forward(conv4_2, x4_1, x4_height, x4_width);
            float ***x4_3 = conv2d_forward(conv4_3, x4_2, x4_height, x4_width);
            float ***x4_mp = max_pooling_2d(x4_3, conv4_2->output_channel, x4_height, x4_height);

            int x5_height = x4_height/2;
            int x5_width = x4_width/2;
            float ***x5_1 = conv2d_forward(conv5_1, x4_mp, x5_height, x5_width);
            float ***x5_2 = conv2d_forward(conv5_2, x5_1, x5_height, x5_width);
            float ***x5_3 = conv2d_forward(conv5_3, x5_2, x5_height, x5_width);
            float ***x5_mp = max_pooling_2d(x5_3, conv5_2->output_channel, x5_height, x5_height);
            // printf("%d\n",x5_height/2);
            float *x_linear = flatto1d(x5_mp, 512, x5_height/2, x5_width/2);
            float *xl_1 = linear_forward(fc1, x_linear);
            float *xl_2 = linear_forward(fc2, xl_1);
            float *xl_3 = linear_forward(fc3, xl_2);

            // int x5_height = x5_height/2;
            
            // int x5_width = x5_width/2;
            // float ***x5_1 = conv2d_forward(conv5_1, x4_mp, x5_height, x5_width);
            // float ***x5_2 = conv2d_forward(conv5_2, x5_1, x5_height, x5_width);
            // float ***x5_3 = conv2d_forward(conv5_3, x5_2, x5_height, x5_width);
            // float ***x5_mp = max_pooling_2d(x5_3, conv5_2->output_channel, x5_height, x5_height);
            // float ***x2 = conv2d_forward(conv2, x1_mp, input_height/2, input_width/2);
            // float ***x2_mp = max_pooling_2d(x2, conv2->output_channel, input_height/2, input_width/2);

            // float *input_linear = flatto1d(x2_mp, 64, (input_height/2)/2, (input_height/2)/2);
            // float *x3 = linear_forward(linear1, input_linear);
            // float *x4 = linear_forward(linear2, x3);

            // free_3d_float_array(input, 1, input_height);
            // free_3d_float_array(x1, conv1->output_channel, input_height);
            // free_3d_float_array(x1_mp, conv1->output_channel, input_height/2);
            // free_3d_float_array(x2, conv2->output_channel, input_height/2);
            // free_3d_float_array(x2_mp, conv2->output_channel, (input_height/2)/2);
            // free(input_linear);
            // free(x3);

            // free(x4);
            
        }
        end = clock();
        double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
        printf("Thời gian thực thi mô hình %d: %f giây\n\n", t, time_taken);
    }

    // srand(time(NULL));
    // clock_t start, end;
    // long a = 0xabcdef12abcdef12;
    // int b = 0xabcdef12;
    // start = clock();
    // for(int i = 0; i< 100000000; i++){
    //     a^a;
    // }
    // end = clock();
    // double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    // printf("Thời gian thực thi mô hình: %f giây\n\n", time_taken);

    // start = clock();
    // for(int i = 0; i< 100000000; i++){
    //     b^b;
    // }
    // end = clock();
    // time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    // printf("Thời gian thực thi mô hình: %f giây\n\n", time_taken);
}
