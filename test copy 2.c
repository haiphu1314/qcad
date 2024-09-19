/**
 * @ Author: Hai Phu
 * @ Email:  haiphu@hcmut.edu.vn
 * @ Create Time: 2024-06-27 16:23:39
 * @ Modified time: 2024-08-31 19:47:08
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
#include <dnnl.h>

#define NUM_TESTCASES 84000
// #define NO_TESTS 1
#define NO_TESTS 1000
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


void simple_conv() {
    // Cấu hình kích thước của input, filter và output
    int batch = 1, channels = 1, height = 5, width = 5;
    int filter_size = 3, stride = 1, padding = 1;

    // Khởi tạo oneDNN engine và stream
    dnnl_engine_t engine;
    dnnl_stream_t stream;
    dnnl_engine_create(&engine, dnnl_cpu, 0);
    dnnl_stream_create(&stream, engine, dnnl_stream_default_flags);

    // Kích thước của input, filter và output
    dnnl_dims_t src_dims = {batch, channels, height, width};
    dnnl_dims_t weights_dims = {channels, channels, filter_size, filter_size};
    dnnl_dims_t bias_dims = {channels};
    dnnl_dims_t dst_dims = {batch, channels, height, width};

    // Tạo memory descriptors
    dnnl_memory_desc_t src_md, weights_md, bias_md, dst_md;
    dnnl_memory_desc_init_by_tag(&src_md, 4, src_dims, dnnl_f32, dnnl_nchw);
    dnnl_memory_desc_init_by_tag(&weights_md, 4, weights_dims, dnnl_f32, dnnl_oihw);
    dnnl_memory_desc_init_by_tag(&bias_md, 1, bias_dims, dnnl_f32, dnnl_x);
    dnnl_memory_desc_init_by_tag(&dst_md, 4, dst_dims, dnnl_f32, dnnl_nchw);

    // Tạo memory objects
    dnnl_memory_t src_mem, weights_mem, bias_mem, dst_mem;
    dnnl_memory_create(&src_mem, &src_md, engine, DNNL_MEMORY_ALLOCATE);
    dnnl_memory_create(&weights_mem, &weights_md, engine, DNNL_MEMORY_ALLOCATE);
    dnnl_memory_create(&bias_mem, &bias_md, engine, DNNL_MEMORY_ALLOCATE);
    dnnl_memory_create(&dst_mem, &dst_md, engine, DNNL_MEMORY_ALLOCATE);

    // Khởi tạo dữ liệu input và filter
    float *src_data = (float *)dnnl_memory_get_data_handle(src_mem);
    float *weights_data = (float *)dnnl_memory_get_data_handle(weights_mem);
    float *bias_data = (float *)dnnl_memory_get_data_handle(bias_mem);
    float *dst_data = (float *)dnnl_memory_get_data_handle(dst_mem);

    for (int i = 0; i < batch * channels * height * width; ++i) src_data[i] = 1.0f;
    for (int i = 0; i < channels * channels * filter_size * filter_size; ++i) weights_data[i] = 1.0f;
    for (int i = 0; i < channels; ++i) bias_data[i] = 0.0f;

    // Cấu hình convolution descriptor
    dnnl_convolution_desc_t conv_desc;
    dnnl_dims_t strides = {stride, stride};
    dnnl_dims_t padding = {padding, padding};
    dnnl_convolution_forward_desc_init(&conv_desc, dnnl_forward, dnnl_convolution_direct,
                                       &src_md, &weights_md, &bias_md, &dst_md,
                                       strides, padding, padding);

    // Tạo primitive descriptor
    dnnl_primitive_desc_t conv_pd;
    dnnl_primitive_desc_create(&conv_pd, &conv_desc, NULL, engine, NULL);

    // Tạo convolution primitive
    dnnl_primitive_t conv;
    dnnl_primitive_create(&conv, conv_pd);

    // Thực thi convolution
    dnnl_exec_arg_t args[4] = {
        {DNNL_ARG_SRC, src_mem},
        {DNNL_ARG_WEIGHTS, weights_mem},
        {DNNL_ARG_BIAS, bias_mem},
        {DNNL_ARG_DST, dst_mem},
    };
    dnnl_primitive_execute(conv, stream, 4, args);

    // Chờ stream hoàn thành
    dnnl_stream_wait(stream);

    // In kết quả
    for (int i = 0; i < batch * channels * height * width; ++i) {
        printf("%f ", dst_data[i]);
        if ((i + 1) % width == 0) printf("\n");
        if ((i + 1) % (width * height) == 0) printf("\n");
    }

    // Giải phóng tài nguyên
    dnnl_primitive_destroy(conv);
    dnnl_primitive_desc_destroy(conv_pd);
    dnnl_memory_destroy(src_mem);
    dnnl_memory_destroy(weights_mem);
    dnnl_memory_destroy(bias_mem);
    dnnl_memory_destroy(dst_mem);
    dnnl_stream_destroy(stream);
    dnnl_engine_destroy(engine);
}


int main()
{
    simple_conv();
    
    // const dnnl_version_t* version = dnnl_version();
    // printf("oneDNN version: %s\n", version->version);
    // printf("Commit hash: %s\n", version->hash);
    return 0;
    // omp_set_num_threads(NUM_CPU);
    // quant_type typ[4] = {FP, TNN, TBN, BNN};
    // int len_typ = 4;
    // double time_FP;
    // for (int t = 0; t < len_typ; t++)
    // // int t = 1;
    // {
    //     int input_height = 28;
    //     int input_width = 28;
    //     layer_node *model = NULL;
    //     conv2d_layer *conv1 = create_conv2d_layer(1, 16, 3, 1, 1, 1, typ[t]);
    //     model = add_layer(model, CONV, "conv1", conv1);
    //     conv2d_layer *conv2 = create_conv2d_layer(6, 16, 5, 1, 0, 1, typ[t]);
    //     model = add_layer(model, CONV, "conv2", conv2);
    //     linear_layer *linear1 = create_linear_layer(16 * 5 * 5, 120, typ[t]);
    //     model = add_layer(model, LINEAR, "linear1", linear1);
    //     linear_layer *linear2 = create_linear_layer(120, 84, typ[t]);
    //     model = add_layer(model, LINEAR, "linear2", linear2);
    //     linear_layer *linear3 = create_linear_layer(84, 10, typ[t]);
    //     model = add_layer(model, LINEAR, "linear3", linear3);

    //     srand(time(NULL));
    //     clock_t start, end;
    //     start = clock();

    //     for (int ct = 0; ct < NO_TESTS; ct++)
    //     {
    //         float ***input = allocate_3d_float_array(1, input_height, input_width);
    //         for (int c = 0; c < 1; c++)
    //         {
    //             for (int h = 0; h < input_height; h++)
    //             {
    //                 for (int w = 0; w < input_width; w++)
    //                 {
    //                     input[c][h][w] = power(-1, w);
    //                 }
    //             }
    //         }
    //         float ***x1 = conv2d_forward(conv1, input, input_height, input_width);
    //         float ***x1_mp = max_pooling_2d(x1, conv1->output_channel, input_height, input_width);
    //         float ***x2 = conv2d_forward(conv2, x1_mp, input_height/2, input_width/2);
    //         float ***x2_mp = max_pooling_2d(x2, conv2->output_channel, input_height/2-4, input_width/2-4);

    //         float *input_linear = flatto1d(x2_mp, 16, 5, 5);
    //         float *x3 = linear_forward(linear1, input_linear);
    //         float *x4 = linear_forward(linear2, x3);
    //         float *x5 = linear_forward(linear3, x4);
            
    //     }
    //     end = clock();
    //     double time_taken = ((double)((end - start))/NUM_CPU) / CLOCKS_PER_SEC;
    //     if(typ[t] == FP){
    //         time_FP = time_taken;
    //     }
    //     printf("Thời gian thực thi mô hình %d: %.6f giây, %.4f\n\n", t, time_taken/NO_TESTS, time_FP/time_taken);
    // }
}
