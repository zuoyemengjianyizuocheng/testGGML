#include "ggml.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, const char ** argv) {
    struct ggml_init_params params = {
        .mem_size   = 128*1024*1024, //申请128MB内存
        .mem_buffer = NULL,
        .no_alloc   = false,
    };

    struct ggml_context * ctx0 = ggml_init(params);     //内部地址以及空间初始化

    //列，行，页
    struct ggml_tensor * t1 = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 10);      //1维，10个
    struct ggml_tensor * t2 = ggml_new_tensor_2d(ctx0, GGML_TYPE_I16, 10, 20);      //2维，10*20，ne=[10,20,1,1]
    struct ggml_tensor * t3 = ggml_new_tensor_3d(ctx0, GGML_TYPE_I32, 10, 20, 30);  //3维,10*20*30,ne=[10,20,30,1]

    GGML_ASSERT(ggml_n_dims(t1) == 1);
    GGML_ASSERT(t1->ne[0]  == 10);
    GGML_ASSERT(t1->nb[1]  == 10*sizeof(float));

    GGML_ASSERT(ggml_n_dims(t2) == 2);
    GGML_ASSERT(t2->ne[0]  == 10);
    GGML_ASSERT(t2->ne[1]  == 20);
    GGML_ASSERT(t2->nb[1]  == 10*sizeof(int16_t));
    GGML_ASSERT(t2->nb[2]  == 10*20*sizeof(int16_t));

    GGML_ASSERT(ggml_n_dims(t3) == 3);
    GGML_ASSERT(t3->ne[0]  == 10);
    GGML_ASSERT(t3->ne[1]  == 20);
    GGML_ASSERT(t3->ne[2]  == 30);
    GGML_ASSERT(t3->nb[1]  == 10*sizeof(int32_t));
    GGML_ASSERT(t3->nb[2]  == 10*20*sizeof(int32_t));
    GGML_ASSERT(t3->nb[3]  == 10*20*30*sizeof(int32_t));

    //t1 t2 t3之间的对齐填充需要32字节，单一数据理论size，t1=4*10 ，t2=2*10*20，t3=4*10*20*30，但如果内存分类不是连续的，那么size大小就不是理论大小
    ggml_print_objects(ctx0);

    ggml_free(ctx0);

    return 0;
}


