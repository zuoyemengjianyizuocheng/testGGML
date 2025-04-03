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
        buf_f16[i] = ggml_fp32_to_fp16((float)i);       //在C中不显示16位float格式，显示会用unsigned short，会有特别大值
    }
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
        memcpy(t_d_2, buf_f16, ggml_nbytes(t_2));


        // ggml_get_rel_pos(ctx, t, a, a)，作用是从t中，第一次取第N=a(从1计数)行为首行，第N-1行为次行一直倒序取a次，第二次取第N=a+1(从1计数)行为首行，第N-1行为次行一直倒序取a次；整个动作重复a次
        struct ggml_tensor * rw = ggml_get_rel_pos(ctx, t, 2, 2);       //一共两次，第一次第一步取t第2行为首行，第二步取第1行为次行，第二次取t第3行为首行，第二步取第2行为次行，取值完成
        struct ggml_tensor * rh = ggml_get_rel_pos(ctx, t_2, 2, 2);
        

        //将相对位置张量的FP16转换为单精度浮点数（FP32）
        struct ggml_tensor * rw_f32 = ggml_cpy(ctx, rw, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 2, 2));       //copy操作，将rw与新建的ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 2, 2)进行类型交换，
        struct ggml_tensor* rh_f32 = ggml_cpy(ctx, rh, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 2, 2));

        struct ggml_tensor * in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 9, 4);
        struct ggml_tensor * out_inplace = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 9, 4);

        float * in_d          = (float*)in->data;
        float * out_inplace_d = (float*)out_inplace->data;
        for (int i = 0; i < ggml_nelements(in); ++i) {
            in_d[i]          = 0.f;
            out_inplace_d[i] = 1.f;
        }

        //使用 ggml_add_rel_pos 函数将相对位置添加到输入张量 in，生成新的输出张量 out
        struct ggml_tensor * out = ggml_add_rel_pos(ctx, in, rw_f32, rh_f32);       //out[i,j] = in[i,j] + rw_f32[i] + rh_f32[j]
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_graph_compute_with_ctx(ctx, gf, 1);

        //struct ggml_cgraph* gf1 = ggml_new_graph(ctx);
        //ggml_build_forward_expand(gf1, rw_f32);
        //ggml_graph_compute_with_ctx(ctx, gf1, 1);
        float* t_d_21 = (float*)out->data;
        for (int k = 0; k < out->ne[2]/* rows */; k++) {
            if (k > 0) {
                printf("\n");
            }
            for (int j = 0; j < out->ne[1]/* rows */; j++) {
                if (j > 0) {
                    printf("\n");
                }

                for (int i = 0; i < out->ne[0]/* cols */; i++) {
                    printf(" %f", t_d_21[k * out->ne[1] * out->ne[0] + j * out->ne[0] + i]);
                }
            }
        }

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
