#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <stdlib.h>
#include <string.h>

struct model {
    struct ggml_context* ctx;       //上下文的内存大小和空间分配的信息
    struct ggml_context* ctx0;
    ggml_backend_t backend;     
    ggml_backend_buffer_t buffer;       //内存
    struct ggml_cgraph* gf;     //计算图
    ggml_gallocr_t allocr;      //内存分配
    uint8_t* buf;
};

struct ggml_context* make_ctx(void) {
    struct ggml_init_params params = {      
        .mem_size = ggml_tensor_overhead() * 3,     //空间大小，是否分配内存
        .mem_buffer = NULL,
        .no_alloc = true,
    };
    return ggml_init(params);       //初始话空间，内存，精度类型转化
}

ggml_backend_t make_backend(void) {     //选定后端
    ggml_backend_t backend = NULL;

#ifdef GGML_USE_CUDA
    backend = ggml_backend_cuda_init(0);
    GGML_ASSERT(backend != NULL);
#endif

    if (!backend) {
        backend = ggml_backend_cpu_init();
    }

    return backend;
}

void model_init(struct model* m) {
    m->ctx = make_ctx();        //初始化状态1
    m->backend = make_backend();        //确定后端

    size_t buf_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();     //内存空间大小
    m->buf = calloc(buf_size, sizeof(uint8_t));     //内存空间
    struct ggml_init_params params0 = {     //内存信息初始化
        .mem_size = buf_size,
        .mem_buffer = m->buf,
        .no_alloc = true,
    };
    m->ctx0 = ggml_init(params0);       //初始化状态2
    m->gf = ggml_new_graph(m->ctx0);    //状态2的计算图
}

void model_alloc(struct model* m) {
    m->buffer = ggml_backend_alloc_ctx_tensors(m->ctx, m->backend);
    m->allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m->backend));
}

void model_compute(struct model* m) {
    ggml_gallocr_alloc_graph(m->allocr, m->gf);
    ggml_backend_graph_compute(m->backend, m->gf);
}

void model_free(struct model* m) {
    ggml_free(m->ctx0);
    free(m->buf);
    ggml_gallocr_free(m->allocr);
    ggml_free(m->ctx);
    ggml_backend_buffer_free(m->buffer);
    ggml_backend_free(m->backend);
}

void check_tensor(struct ggml_tensor* t,
                  const float* expected_t_d,
                  const int ne0,
                  const int ne1,
                  const int ne2) {
    GGML_ASSERT(t->ne[0] == ne0);       //维度判定
    GGML_ASSERT(t->ne[1] == ne1);
    GGML_ASSERT(t->ne[2] == ne2);
    const size_t bsize = ggml_nbytes(t);
    if (t->type == GGML_TYPE_F32) {
        float* buffer = malloc(bsize);
        ggml_backend_tensor_get(t, buffer, 0, bsize);
        for (int i = 0; i < bsize / sizeof(float); ++i) {
            float expected = expected_t_d[i];       //buffer数值比较
            float actual = buffer[i];
            if (expected != actual) {
                printf("expected %.1f, got %.1f\n", expected, actual);
            }
            GGML_ASSERT(expected == actual);
        }
        free(buffer);
    } else if (t->type == GGML_TYPE_F16) {
        ggml_fp16_t* buffer = malloc(bsize);
        ggml_backend_tensor_get(t, buffer, 0, bsize);
        for (int i = 0; i < bsize / sizeof(ggml_fp16_t); ++i) {
            float expected = expected_t_d[i];
            float actual = ggml_fp16_to_fp32(buffer[i]);
            if (expected != actual) {
                printf("expected %.1f, got %.1f\n", expected, actual);
            }
            GGML_ASSERT(expected == actual);
        }
        free(buffer);
    //} else if (t->type == GGML_TYPE_BF16) {
    //    ggml_bf16_t* buffer = malloc(bsize);
    //    ggml_backend_tensor_get(t, buffer, 0, bsize);
    //    for (int i = 0; i < bsize / sizeof(ggml_bf16_t); ++i) {
    //        float expected = expected_t_d[i];
    //        float actual = ggml_bf16_to_fp32(buffer[i]);
    //        if (expected != actual) {
    //            printf("expected %.1f, got %.1f\n", expected, actual);
    //        }
    //        GGML_ASSERT(expected == actual);
    //    }
    //    free(buffer);
    } else {
        GGML_ABORT("unknown type");
    }
}

void test_cont(void) {
    float buf_f32[] = {1.0, 2.0};       //新建float32类型
    ggml_fp16_t buf_f16[] = {ggml_fp32_to_fp16(buf_f32[0]), ggml_fp32_to_fp16(buf_f32[1])};     //将float32转化为float16
    ggml_bf16_t buf_bf16[] = {ggml_fp32_to_bf16(buf_f32[0]), ggml_fp32_to_bf16(buf_f32[1])};    //将float32转化为bf16

    float expected_out[] = {1.0, 2.0};

    struct model m;         //新建模型
    model_init(&m);         //初始化模型M中的参数

    struct ggml_tensor* in_1 = ggml_new_tensor_1d(m.ctx, GGML_TYPE_F32, 2);     //以m.ctx为上下文，以F32格式，1维形式，2个元素生成张量in_1
    struct ggml_tensor* in_2 = ggml_new_tensor_1d(m.ctx, GGML_TYPE_F16, 2);     //
    //struct ggml_tensor* in_3 = ggml_new_tensor_1d(m.ctx, GGML_TYPE_BF16, 2);

    model_alloc(&m);        //内存分配

    ggml_backend_tensor_set(in_1, buf_f32, 0, ggml_nbytes(in_1));       //后端张量设置，将buf_f32的数据写入in_1中
    ggml_backend_tensor_set(in_2, buf_f16, 0, ggml_nbytes(in_2));
    //ggml_backend_tensor_set(in_3, buf_bf16, 0, ggml_nbytes(in_3));

    struct ggml_tensor* out_1 = ggml_cont(m.ctx0, ggml_transpose(m.ctx0, in_1));    //ggml_transpose转置in_1，然后进行ggml_cont组合，生成out_1
    struct ggml_tensor* out_2 = ggml_cont(m.ctx0, ggml_transpose(m.ctx0, in_2));
    //struct ggml_tensor* out_3 = ggml_cont(m.ctx0, ggml_transpose(m.ctx0, in_3));

    ggml_build_forward_expand(m.gf, out_1);     //前向计算，构建扩展图
    ggml_build_forward_expand(m.gf, out_2);
    //ggml_build_forward_expand(m.gf, out_3);

    model_compute(&m);      //计算

    check_tensor(out_1, expected_out, 1, 2, 1);     //检查数值以及维度是否一致
    check_tensor(out_2, expected_out, 1, 2, 1);
    //check_tensor(out_3, expected_out, 1, 2, 1);

    model_free(&m);
}

int main(int argc, const char* argv[]) {
    test_cont();
    return 0;
}
