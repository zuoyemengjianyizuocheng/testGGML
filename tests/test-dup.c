#include "ggml.h"
#include "ggml-cpu.h"

#include <stdio.h>
#include <stdlib.h>

void arange(struct ggml_tensor* tensor) {
    GGML_ASSERT(ggml_is_contiguous(tensor));
    for (int i = 0; i < ggml_nelements(tensor); ++i) {
        ggml_set_i32_1d(tensor, i, i);
    }
}

void dup_to(struct ggml_tensor* src, struct ggml_tensor* dst) {
    GGML_ASSERT(dst->op == GGML_OP_VIEW);
    GGML_ASSERT(ggml_nelements(src) == ggml_nelements(dst));
    dst->op = GGML_OP_DUP;
    dst->src[0] = src;
}

bool can_dup(enum ggml_type src_type, enum ggml_type dst_type) {
    if (src_type == dst_type) return true;
    if (src_type == GGML_TYPE_F32 && ggml_get_type_traits_cpu(dst_type)->from_float) return true;       // 如果源类型是浮点数，并且目标类型支持从浮点数转换，返回 true
    if (dst_type == GGML_TYPE_F32 && ggml_get_type_traits    (src_type)->to_float) return true;     // 如果目标类型是浮点数，并且源类型支持转换为浮点数，返回 true

    return false;
}

int main(int argc, const char ** argv) {
    struct ggml_init_params params = {
        .mem_size   = 128*1024*1024,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };

    enum ggml_type type[4] = {GGML_TYPE_I16, GGML_TYPE_I32, GGML_TYPE_F16, GGML_TYPE_F32};
    for (int i = 0; i < 4; ++i) {
        enum ggml_type src_type = type[i];
        for (int j = 0; j < 4; ++j) {
            enum ggml_type dst_type = type[j];
            if (!can_dup(src_type, dst_type)) continue;     //类型保持一致
            printf("Testing dup on %s -> %s copy\n", ggml_type_name(src_type), ggml_type_name(dst_type));

            struct ggml_context * ctx = ggml_init(params);

            struct ggml_tensor * src = ggml_new_tensor_2d(ctx, src_type, 10, 11);
            arange(src);        //将 src 张量初始化为一个包含从 0 到 (10*11-1) 的连续整数的数组
            struct ggml_tensor * dst = ggml_new_tensor_2d(ctx, dst_type, 10, 11);
            ggml_set_i32(dst, 0);       //将 dst 张量的所有元素初始化为整数值 0

            // 2nd-row: [20, 21, ..., 29]
            struct ggml_tensor * src_cont = ggml_view_1d(ctx, src, 10, src->nb[1] * 2);     //创建1维视图，源自tensor src,该视图有10个元素，每个元素大小为nb[1] * 2，对应于目标张量src的第 2 行

            // 3rd-col: [03, 13, ..., 93]
            struct ggml_tensor * src_stride = ggml_view_2d(ctx, src, 1, 10, src->nb[1], src->nb[0] * 3);        //创建2维视图，源自src，数量=1，视图有10行，每行src->nb[1]列，步长为nb[0] * 3

            struct ggml_tensor * dst_cont_1 = ggml_view_1d(ctx, dst, 10, dst->nb[1] * 5); // 5nd-row
            struct ggml_tensor * dst_cont_2 = ggml_view_1d(ctx, dst, 10, dst->nb[1] * 6); // 6rd-row

            struct ggml_tensor * dst_stride_1 = ggml_view_2d(ctx, dst, 1, 10, dst->nb[1], dst->nb[0] * 7); // 7th-col  //创建2维视图，源自dst，数量=1，视图有10行，每行dst->nb[1]]列，步长为nb[0] * 7，对应于目标张量dst的第 7 列
            struct ggml_tensor * dst_stride_2 = ggml_view_2d(ctx, dst, 1, 10, dst->nb[1], dst->nb[0] * 8); // 8th-col

            struct ggml_cgraph * gf = ggml_new_graph(ctx);

            //将源张量的数据复制到目标张量的不同视图中
            dup_to(src_cont,   dst_cont_1);
            dup_to(src_stride, dst_cont_2);
            dup_to(src_cont,   dst_stride_1);
            dup_to(src_stride, dst_stride_2);

            ggml_build_forward_expand(gf, dst_cont_1);
            ggml_build_forward_expand(gf, dst_cont_2);
            ggml_build_forward_expand(gf, dst_stride_1);
            ggml_build_forward_expand(gf, dst_stride_2);

            ggml_graph_compute_with_ctx(ctx, gf, 1);

            // src_cont -> dst_cont_1
            GGML_ASSERT(ggml_get_i32_1d(dst, 49) == 0);
            GGML_ASSERT(ggml_get_i32_1d(dst, 50) == 20);
            GGML_ASSERT(ggml_get_i32_1d(dst, 51) == 21);
            GGML_ASSERT(ggml_get_i32_1d(dst, 52) == 22);
            GGML_ASSERT(ggml_get_i32_1d(dst, 59) == 29);

            // src_stride -> dst_cont_2
            GGML_ASSERT(ggml_get_i32_1d(dst, 60) == 3);
            GGML_ASSERT(ggml_get_i32_1d(dst, 61) == 13);
            GGML_ASSERT(ggml_get_i32_1d(dst, 62) == 23);
            GGML_ASSERT(ggml_get_i32_1d(dst, 69) == 93);
            GGML_ASSERT(ggml_get_i32_1d(dst, 70) == 0);

            // src_cont -> dst_stride_1
            GGML_ASSERT(ggml_get_i32_1d(dst, 6)   == 0);
            GGML_ASSERT(ggml_get_i32_1d(dst, 7)   == 20);
            GGML_ASSERT(ggml_get_i32_1d(dst, 17)  == 21);
            GGML_ASSERT(ggml_get_i32_1d(dst, 27)  == 22);
            GGML_ASSERT(ggml_get_i32_1d(dst, 97)  == 29);
            GGML_ASSERT(ggml_get_i32_1d(dst, 107) == 0);

            // src_stride -> dst_stride_2
            GGML_ASSERT(ggml_get_i32_1d(dst, 8)   == 03);
            GGML_ASSERT(ggml_get_i32_1d(dst, 18)  == 13);
            GGML_ASSERT(ggml_get_i32_1d(dst, 28)  == 23);
            GGML_ASSERT(ggml_get_i32_1d(dst, 98)  == 93);
            GGML_ASSERT(ggml_get_i32_1d(dst, 108) == 0);

            ggml_free(ctx);
        }
    }

    return 0;
}
