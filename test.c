#include <stdio.h>
#include <stdlib.h>
#include <dnnl.h>

#define CHECK(f)                                          \
    do {                                                  \
        dnnl_status_t s = f;                              \
        if (s != dnnl_success) {                          \
            printf("Error: %d\n", s);                     \
            exit(1);                                      \
        }                                                 \
    } while (0)

void conv2d_oneDNN(const float *input, const float *weights, float *output,
                   int input_channels, int output_channels, int input_height, int input_width,
                   int kernel_size, int stride, int padding, int dilation)
{
    dnnl_engine_t engine;
    dnnl_stream_t stream;
    CHECK(dnnl_engine_create(&engine, dnnl_cpu, 0));
    CHECK(dnnl_stream_create(&stream, engine, dnnl_stream_default_flags));

    int output_height = (input_height - kernel_size + 2 * padding) / stride + 1;
    int output_width = (input_width - kernel_size + 2 * padding) / stride + 1;

    dnnl_memory_desc_t src_md, weights_md, dst_md;
    dnnl_memory_t src_mem, weights_mem, dst_mem;

    dnnl_dim_t src_dims[4] = {1, input_channels, input_height, input_width};
    dnnl_dim_t weights_dims[4] = {output_channels, input_channels, kernel_size, kernel_size};
    dnnl_dim_t dst_dims[4] = {1, output_channels, output_height, output_width};

    CHECK(dnnl_memory_desc_init_by_tag(&src_md, 4, src_dims, dnnl_f32, dnnl_nchw));
    CHECK(dnnl_memory_desc_init_by_tag(&weights_md, 4, weights_dims, dnnl_f32, dnnl_oihw));
    CHECK(dnnl_memory_desc_init_by_tag(&dst_md, 4, dst_dims, dnnl_f32, dnnl_nchw));

    CHECK(dnnl_memory_create(&src_mem, &src_md, engine, DNNL_MEMORY_ALLOCATE));
    CHECK(dnnl_memory_create(&weights_mem, &weights_md, engine, DNNL_MEMORY_ALLOCATE));
    CHECK(dnnl_memory_create(&dst_mem, &dst_md, engine, DNNL_MEMORY_ALLOCATE));

    CHECK(dnnl_memory_set_data_handle(src_mem, (void*)input));
    CHECK(dnnl_memory_set_data_handle(weights_mem, (void*)weights));

    dnnl_convolution_desc_t conv_desc;
    dnnl_dim_t strides[2] = {stride, stride};
    dnnl_dim_t padding_dims[2] = {padding, padding};

    // Không truyền dilation vào nữa vì dilation được gộp vào strides.
    CHECK(dnnl_convolution_forward_desc_init(&conv_desc, dnnl_forward_inference,
                                             dnnl_convolution_direct, &src_md, &weights_md, NULL, &dst_md,
                                             strides, padding_dims, padding_dims));

    dnnl_primitive_desc_t conv_pd;
    CHECK(dnnl_primitive_desc_create(&conv_pd, &conv_desc, NULL, engine, NULL));

    dnnl_primitive_t conv;
    CHECK(dnnl_primitive_create(&conv, conv_pd));

    dnnl_exec_arg_t conv_args[3] = {
        {DNNL_ARG_SRC, src_mem},
        {DNNL_ARG_WEIGHTS, weights_mem},
        {DNNL_ARG_DST, dst_mem},
    };

    CHECK(dnnl_primitive_execute(conv, stream, 3, conv_args));

    CHECK(dnnl_memory_get_data_handle(dst_mem, (void**)&output));

    CHECK(dnnl_primitive_desc_destroy(conv_pd));
    CHECK(dnnl_primitive_destroy(conv));
    CHECK(dnnl_memory_destroy(src_mem));
    CHECK(dnnl_memory_destroy(weights_mem));
    CHECK(dnnl_memory_destroy(dst_mem));
    CHECK(dnnl_stream_destroy(stream));
    CHECK(dnnl_engine_destroy(engine));
}

int main() {
    int input_channels = 3, output_channels = 16;
    int input_height = 32, input_width = 32;
    int kernel_size = 3, stride = 1, padding = 1, dilation = 1;

    float input[input_channels * input_height * input_width];
    float weights[output_channels * input_channels * kernel_size * kernel_size];
    int output_height = (input_height - kernel_size + 2 * padding) / stride + 1;
    int output_width = (input_width - kernel_size + 2 * padding) / stride + 1;
    float output[output_channels * output_height * output_width];

    conv2d_oneDNN(input, weights, output, input_channels, output_channels,
                  input_height, input_width, kernel_size, stride, padding, dilation);

    printf("Convolution completed successfully ahihi.\n");

    return 0;

}
