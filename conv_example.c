#include <stdio.h>
#include <stdlib.h>
#include <dnnl.h>

// Hàm tiện ích kiểm tra lỗi
void check_status(dnnl_status_t status, const char* msg) {
    if (status != dnnl_success) {
        fprintf(stderr, "%s\n", msg);
        exit(1);
    }
}

int main() {
    // Step 1: Khởi tạo engine và stream
    dnnl_engine_t engine;
    dnnl_stream_t stream;
    dnnl_status_t status;

    status = dnnl_engine_create(&engine, dnnl_cpu, 0);
    check_status(status, "Error creating engine");

    status = dnnl_stream_create(&stream, engine, dnnl_stream_default_flags);
    check_status(status, "Error creating stream");

    // Step 2: Định nghĩa các tensor
    // Giả sử input là 1 ảnh 1x1x4x4, kernel là 1x1x3x3, và stride là 1x1
    int batch = 1, channels = 1, height = 4, width = 4;
    int kernel_h = 3, kernel_w = 3;

    dnnl_memory_desc_t src_md, weights_md, dst_md;
    dnnl_memory_t src_mem, weights_mem, dst_mem;

    status = dnnl_memory_desc_init_by_tag(&src_md, 4, (dnnl_dims_t){batch, channels, height, width}, 
                                          dnnl_f32, dnnl_nchw);
    check_status(status, "Error creating source memory descriptor");

    status = dnnl_memory_desc_init_by_tag(&weights_md, 4, (dnnl_dims_t){channels, channels, kernel_h, kernel_w}, 
                                          dnnl_f32, dnnl_oihw);
    check_status(status, "Error creating weights memory descriptor");

    status = dnnl_memory_desc_init_by_tag(&dst_md, 4, (dnnl_dims_t){batch, channels, height - kernel_h + 1, width - kernel_w + 1}, 
                                          dnnl_f32, dnnl_nchw);
    check_status(status, "Error creating destination memory descriptor");

    status = dnnl_memory_create(&src_mem, &src_md, engine, DNNL_MEMORY_ALLOCATE);
    check_status(status, "Error creating source memory");

    status = dnnl_memory_create(&weights_mem, &weights_md, engine, DNNL_MEMORY_ALLOCATE);
    check_status(status, "Error creating weights memory");

    status = dnnl_memory_create(&dst_mem, &dst_md, engine, DNNL_MEMORY_ALLOCATE);
    check_status(status, "Error creating destination memory");

    // Step 3: Thiết lập và thực thi lớp convolution
    dnnl_convolution_desc_t conv_desc;
    dnnl_primitive_desc_t conv_pd;
    dnnl_primitive_t conv;

    dnnl_dims_t strides = {1, 1};
    dnnl_dims_t padding = {0, 0};

    status = dnnl_convolution_forward_desc_init(&conv_desc, dnnl_forward, 
                                                dnnl_convolution_direct, 
                                                &src_md, &weights_md, NULL, &dst_md, 
                                                strides, padding, padding);
    check_status(status, "Error creating convolution descriptor");

    status = dnnl_primitive_desc_create(&conv_pd, &conv_desc, NULL, engine, NULL);
    check_status(status, "Error creating convolution primitive descriptor");

    status = dnnl_primitive_create(&conv, conv_pd);
    check_status(status, "Error creating convolution primitive");

    dnnl_exec_arg_t args[3] = {
        {DNNL_ARG_SRC, src_mem},
        {DNNL_ARG_WEIGHTS, weights_mem},
        {DNNL_ARG_DST, dst_mem}
    };

    status = dnnl_primitive_execute(conv, stream, 3, args);
    check_status(status, "Error executing convolution");

    status = dnnl_stream_wait(stream);
    check_status(status, "Error waiting for stream");

    // Step 4: Giải phóng tài nguyên
    dnnl_primitive_desc_destroy(conv_pd);
    dnnl_primitive_destroy(conv);
    dnnl_memory_destroy(src_mem);
    dnnl_memory_destroy(weights_mem);
    dnnl_memory_destroy(dst_mem);
    dnnl_stream_destroy(stream);
    dnnl_engine_destroy(engine);

    printf("Convolution completed successfully.\n");
    return 0;
}
