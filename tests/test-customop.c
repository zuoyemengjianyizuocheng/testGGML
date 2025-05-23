#include "ggml.h"
#include "ggml-cpu.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#if defined(_WIN32)
#include <windows.h>
typedef volatile LONG atomic_int;
static LONG atomic_fetch_add(atomic_int * ptr, LONG inc) {
    return InterlockedExchangeAdd(ptr, inc);
}
#else
#include <stdatomic.h>
#endif

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

struct ggml_context * make_ctx(void) {
    struct ggml_init_params params = {
        /*.mem_size   =*/ 1 * 1024 * 1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };

    return ggml_init(params);
}

char g_userdata[] = "ggml";
atomic_int g_custom1_count = 0;
atomic_int g_custom2_count = 0;
atomic_int g_custom3_count = 0;

void custom1(struct ggml_tensor * dst , const struct ggml_tensor * a, int ith, int nth, void * userdata) {
    // check that the userdata is correct
    assert(userdata == NULL);
    assert(ggml_are_same_shape(dst, a));

    //atomic_fetch_add 函数会获取一个原子类型的当前值，将给定的递增值添加到这个原子值上，然后返回原始的未递增的值。
    // 整个操作是原子性的，即从读取（返回）值到此函数修改值的时刻，该值不受其他线程的影响
    atomic_fetch_add(&g_custom1_count, 1);      //指定原子加法

    const float * a_data = ggml_get_data_f32(a);        //返回tensor a的data指针给指针a_data
    float * dst_data = ggml_get_data_f32(dst);

    //this assumes that the tensors are contiguous 张量连续
    assert(ggml_is_contiguous(dst));
    assert(ggml_is_contiguous(a));

    // parallelize by elements 按元素并行化
    const int ne = (int)ggml_nelements(dst);        //元素个数
    const int dr = (ne + nth - 1) / nth;
    const int ie0 = dr * ith;
    const int ie1 = MIN(ie0 + dr, ne);

    for (int i = ie0; i < ie1; ++i) {
        dst_data[i] = a_data[i] * 2;        
    }
}

void custom2(struct ggml_tensor * dst , const struct ggml_tensor * a, const struct ggml_tensor * b, int ith, int nth, void * userdata) {
    // check that the userdata is correct
    assert(userdata == g_userdata);
    assert(strcmp(userdata, "ggml") == 0);
    assert(ggml_are_same_shape(dst, a));
    assert(ggml_are_same_shape(dst, b));

    atomic_fetch_add(&g_custom2_count, 1); 

    const float * a_data = ggml_get_data_f32(a);
    const float * b_data = ggml_get_data_f32(b);
    float * dst_data = ggml_get_data_f32(dst);

    // parallelize by rows
    const int nr = (int)ggml_nrows(dst);
    // number of rows per thread  每个线程的行数
    const int dr = (nr + nth - 1) / nth;
    // row range for this thread 此线程的行范围
    const int ir0 = dr * ith;
    const int ir1 = MIN(ir0 + dr, nr);

    // number of columns 列数
    const int nc = (int)dst->ne[0];

    // this assumes that the tensors are contiguous 连续性检测
    assert(ggml_is_contiguous(dst));
    assert(ggml_is_contiguous(a));
    assert(ggml_is_contiguous(b));

    for (int ir = ir0; ir < ir1; ++ir) {
        for (int ic = 0; ic < nc; ++ic) {
            const int i = ir * nc + ic;
            dst_data[i] = a_data[i] + b_data[i]; 
        }
    }
}

void custom3(struct ggml_tensor * dst , const struct ggml_tensor * a, const struct ggml_tensor * b, const struct ggml_tensor * c, int ith, int nth, void * userdata) {
    // check that the userdata is correct
    assert(userdata == g_userdata);
    assert(strcmp(userdata, "ggml") == 0);
    assert(ggml_are_same_shape(dst, a));
    assert(ggml_are_same_shape(dst, b));
    assert(ggml_are_same_shape(dst, c));

    atomic_fetch_add(&g_custom3_count, 1);

    const float * a_data = ggml_get_data_f32(a);
    const float * b_data = ggml_get_data_f32(b);
    const float * c_data = ggml_get_data_f32(c);
    float * dst_data = ggml_get_data_f32(dst);

    // dont parallelize
    assert(ith == 0);

    // number of elements
    const int ne = (int)ggml_nelements(dst);

    // this assumes that the tensors are contiguous
    assert(ggml_is_contiguous(dst));
    assert(ggml_is_contiguous(a));
    assert(ggml_is_contiguous(b));
    assert(ggml_is_contiguous(c));

    for (int i = 0; i < ne; ++i) {
        dst_data[i] = a_data[i] + b_data[i] + c_data[i];
    }
}

int main(int argc, const char** argv) {

    float buf1_f32[1024];
    for (int i = 0; i < 1024; ++i) {
        buf1_f32[i] = (float)(i + 1);
    }
    float buf2_f32[1024];
    for (int i = 0; i < 1024; ++i) {
        buf2_f32[i] = (float)(i + 1) * 2;
    }
    float buf3_f32[1024];
    for (int i = 0; i < 1024; ++i) {
        buf3_f32[i] = (float)(i + 1) * 3;
    }

    // map_custom1
    // 2 tasks, no userdata, parallelized by elements
    {
        struct ggml_context * ctx = make_ctx();     //新建结构，初始化上下文的空间以及内存分配情况
        struct ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 10, 2);     //在ctx中分配一个tensor t的空间，维度为10*2，类型F32
        memcpy(t->data, buf1_f32, ggml_nbytes(t));      //tensor t的data指针指向buf1_f32

        struct ggml_tensor * m1 = ggml_map_custom1(ctx, t, custom1, 2, NULL);       // 自定义操作custom1 ， 执行2次，没有userdata，按元素并行化

        struct ggml_cgraph * graph = ggml_new_graph(ctx);       //创建图
        ggml_build_forward_expand(graph, m1);       //构建图计算逻辑，使用映射的自定义操作m1构建图的前向传递

        ggml_graph_compute_with_ctx(ctx, graph, 4);     //使用4个线程计算上面上下文的图。

        const float * output = ggml_get_data_f32(m1);       //获取计算完后的m1的data指针

        for (int i = 0; i < ggml_nelements(m1); ++i) {
            assert(output[i] == buf1_f32[i] * 2);
        }
        assert(g_custom1_count == 2);

        ggml_free(ctx);
    }

    // map_custom2
    // max tasks (4), userdata, parallelized by rows
    {
        struct ggml_context * ctx = make_ctx();
        struct ggml_tensor * t1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 10, 2);
        memcpy(t1->data, buf1_f32, ggml_nbytes(t1));
        struct ggml_tensor * t2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 10, 2);
        memcpy(t2->data, buf2_f32, ggml_nbytes(t2));

        //将自定义操作custom2映射到最大并行度的输入张量t1和t2， userdata "ggml"，执行GGML_N_TASKS_MAX次
        struct ggml_tensor * m2 = ggml_map_custom2(ctx, t1, t2, custom2, GGML_N_TASKS_MAX, g_userdata);

        struct ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, m2);

        ggml_graph_compute_with_ctx(ctx, graph, 4);     //4线程

        const float * output = ggml_get_data_f32(m2);

        for (int i = 0; i < ggml_nelements(m2); ++i) {
            assert(output[i] == buf1_f32[i] + buf2_f32[i]);
        }

        assert(g_custom2_count == 4);

        ggml_free(ctx);
    }

    // map_custom3
    // 1 task, userdata, not parallelized
    {
        struct ggml_context * ctx = make_ctx();
        struct ggml_tensor * t1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 10, 2);
        memcpy(t1->data, buf1_f32, ggml_nbytes(t1));
        struct ggml_tensor * t2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 10, 2);
        memcpy(t2->data, buf2_f32, ggml_nbytes(t2));
        struct ggml_tensor * t3 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 10, 2);
        memcpy(t3->data, buf3_f32, ggml_nbytes(t3));

        struct ggml_tensor * m3 = ggml_map_custom3(ctx, t1, t2, t3, custom3, 1, g_userdata);        //task==1

        struct ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, m3);

        ggml_graph_compute_with_ctx(ctx, graph, 4);     //4线程

        const float * output = ggml_get_data_f32(m3);

        for (int i = 0; i < ggml_nelements(m3); ++i) {
            assert(output[i] == buf1_f32[i] + buf2_f32[i] + buf3_f32[i]);
        }

        assert(g_custom3_count == 1);

        ggml_free(ctx);
    }


    return 0;
}
