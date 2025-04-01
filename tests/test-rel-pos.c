#include "ggml.h"
#include "ggml-cpu.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

struct ggml_context* make_ctx(void) {
    struct ggml_init_params params = {
        .mem_size = 2 * 1024 * 1024,
    };

    return ggml_init(params);
}

void check_tensor(struct ggml_tensor * t, float * expected_t_d, int ne0, int ne1, int ne2) {
    GGML_ASSERT(t->type == GGML_TYPE_F32);
    GGML_ASSERT(t->ne[0] == ne0);
    GGML_ASSERT(t->ne[1] == ne1);
    GGML_ASSERT(t->ne[2] == ne2);
    for (int i2 = 0; i2 < ne2; ++i2) {
        for (int i1 = 0; i1 < ne1; ++i1) {
            for (int i0 = 0; i0 < ne0; ++i0) {
                float expected = *(expected_t_d + i2 * ne1 * ne0 + i1 * ne0 + i0);
                float actual = ggml_get_data_f32(t)[i2 * ne1 * ne0 + i1 * ne0 + i0];
                GGML_ASSERT(expected == actual);
            }
        }
    }
}

int main(int argc, const char** argv) {
    ggml_fp16_t buf_f16[1024];
    for (int i = 0; i < 1024; ++i) {
        buf_f16[i] = ggml_fp32_to_fp16((float)i);
    }
    printf("%f", buf_f16[2]);
    float expected_out[4][9] = {
        { 8.0, 9.0, 10.0, 9.0, 10.0, 11.0, 10.0, 11.0, 12.0 },
        { 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 4.0, 5.0, 6.0 },
        { 14.0, 15.0, 16.0, 15.0, 16.0, 17.0, 16.0, 17.0, 18.0 },
        { 8.0, 9.0, 10.0, 9.0, 10.0, 11.0, 10.0, 11.0, 12.0 },
    };

    {
        struct ggml_context * ctx = make_ctx();


        struct ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 3, 3);
        ggml_fp16_t* t_d = (ggml_fp16_t*)t->data;       //指针传递
        memcpy(t_d, buf_f16, ggml_nbytes(t));       //t_d指针指向的数值赋值，长度为3*3,同样指针指向的t->data也被赋值

        struct ggml_tensor * t_2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 3, 3);
        ggml_fp16_t* t_d_2 = (ggml_fp16_t*)t_2->data;       
        memcpy(t_d_2, buf_f16 + 1, ggml_nbytes(t_2));


        //实际计算是rw：从 t 中提取相对位置张量，形状为 (3, 2, 2)。
        //计算逻辑：对t，取t.ne[0]次，取 t 中[i:i + 2, j : j + 2] 的 2x2 子矩阵。当先移动j的位置，再移动i，例如第1次，则是t在位置[0,0]的子矩阵构成的二维数组，当出现越界时，则对越界进行补偿，从下一行的开始位置进行补偿
        struct ggml_tensor * rw = ggml_get_rel_pos(ctx, t, 2, 2);       //此处取相对位置张量的时候并未取值，是将t作为src[0]记录到tensor结构中
        struct ggml_tensor * rh = ggml_get_rel_pos(ctx, t_2, 2, 2);

        //将相对位置张量的FP16转换为单精度浮点数（FP32），值依然未取值
        struct ggml_tensor * rw_f32 = ggml_cpy(ctx, rw, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 2, 2));       //copy操作，将rw与新建的ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 2, 2)进行交换，
        struct ggml_tensor* rh_f32 = ggml_cpy(ctx, rh, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 2, 2));
        

        struct ggml_tensor * in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 9, 4);
        struct ggml_tensor * out_inplace = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 9, 4);
        float * in_d          = (float*)in->data;
        float * out_inplace_d = (float*)out_inplace->data;
        for (int i = 0; i < ggml_nelements(in); ++i) {
            in_d[i]          = 1.f;
            out_inplace_d[i] = 1.f;
        }

        //使用 ggml_add_rel_pos 函数将相对位置添加到输入张量 in，生成新的输出张量 out
        struct ggml_tensor * out = ggml_add_rel_pos(ctx, in, rw_f32, rh_f32);       //out[i,j] = in[i,j] + rw_f32[i] + rh_f32[j]
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_graph_compute_with_ctx(ctx, gf, 1);
        float* result_data1 = (float*)out->data;
        

        //使用 ggml_add_rel_pos_inplace 函数在原地修改 out_inplace
        out_inplace = ggml_add_rel_pos_inplace(ctx, out_inplace, rw_f32, rh_f32);       //out[i,j] = 1 + rw_f32[i] + rh_f32[j]
        struct ggml_cgraph * gf_2 = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf_2, out_inplace);
        ggml_graph_compute_with_ctx(ctx, gf_2, 1);

        check_tensor(out, (float*)expected_out, 9, 4, 1);
        check_tensor(out_inplace, (float*)expected_out, 9, 4, 1);
    }

    return 0;
}
