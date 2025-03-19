#include "ggml.h"
#include "ggml-cpu.h"
#include <string.h>
#include <stdio.h>

int main(void) {
    // initialize data of matrices to perform matrix multiplication
    const int rows_A = 4, cols_A = 2;
    float matrix_A[4 * 2] = {
        2, 8,
        5, 1,
        4, 2,
        8, 6
    };
    const int rows_B = 3, cols_B = 2;
    float matrix_B[3 * 2] = {
        10, 5,
        9, 9,
        5, 4
    };

    // 1. Allocate分配 `ggml_context` to store tensor data
    // Calculate计算 the size needed to allocate
    size_t ctx_size = 0;
    ctx_size += rows_A * cols_A * ggml_type_size(GGML_TYPE_F32); // tensor a matrixsize
    ctx_size += rows_B * cols_B * ggml_type_size(GGML_TYPE_F32); // tensor b matrixsize
    ctx_size += rows_A * rows_B * ggml_type_size(GGML_TYPE_F32); // result matrix multiply 矩阵相乘
    ctx_size += 3 * ggml_tensor_overhead(); // metadata for 3 tensors  3个张量的元数据
    ctx_size += ggml_graph_overhead(); // compute graph         计算缓存
    ctx_size += 1024; // some overhead (exact calculation omitted for simplicity) 一些开销（为简单起见省略了精确计算）

    // Allocate 分配`ggml_context` to store tensor data
    struct ggml_init_params params = {
        /*.mem_size =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc =*/ false,
    };

    //初始化内部空间，时钟，时间等信息
    struct ggml_context* ctx = ggml_init(params); 

    // 2. Create tensors and set data
    struct ggml_tensor* tensor_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols_A, rows_A);  //创建tensor数据结构，内置1个结构体ctx，ne表示该数据的维度分布情况，类型定义GGML_TYPE_F32，从初始的ctx空间中分配部分内存存储数据二维数据A，列*行
    struct ggml_tensor* tensor_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols_B, rows_B);  // ggml_new_tensor_2d()会从ctx0的mem_buffer中分配一块内存来保存 tensor x
    memcpy(tensor_a->data, matrix_A, ggml_nbytes(tensor_a));   //赋值
    memcpy(tensor_b->data, matrix_B, ggml_nbytes(tensor_b));


    // 3. Create a `ggml_cgraph` for mul_mat operation为mul_mat操作创建一个‘ ggml_cgraph ’类型，构建计算图谱空间分配等基本信息
    struct ggml_cgraph* gf = ggml_new_graph(ctx);

    // result = a*b^T
    // Pay attention: ggml_mul_mat(A, B) ==> B will be transposed internally
    // the result is transposed  结果= a*b^T //注意：ggml_mul_mat(A， B) ==>；B会在内部转置 //结果被转置
    struct ggml_tensor* result = ggml_mul_mat(ctx, tensor_a, tensor_b);   //此处的result也是未计算状态，仅仅是将待计算变量放入result的数据结构中，等待前向计算开始计算

    // Mark the "result" tensor to be computed 标记“结果”。待计算张量  //构建前向计算图 ，并未进行计算
    ggml_build_forward_expand(gf, result);
    float* result_data2 = (float*)result->data;
    for (int i = 0; i < 12/* cols */; i++) {
        printf(" %.2f\n", result_data2[i]);
    }

    // 4. Run the computation
    int n_threads = 1; // Optional: number of threads to perform some operations with multi-threading 可选：执行某些多线程操作的线程数,执行前向计算
    ggml_graph_compute_with_ctx(ctx, gf, n_threads);  //开启线程计算，此刻才开始正式计算
    float* result_data1 = (float*)result->data;  //tensor数据结构中的data指针指向真实data的位置，所以这里是直接赋值指针
    for (int i = 0; i < 12/* cols */; i++) {
        printf(" %.2f\n", result_data1[i]);
    }

    // 5. Retrieve results (output tensors)
    float* result_data = (float*)result->data;
    printf("mul mat (%d x %d) (transposed result):\n[", (int)result->ne[0], (int)result->ne[1]);
    for (int j = 0; j < result->ne[1]/* rows */; j++) {
        if (j > 0) {
            printf("\n");
        }

        for (int i = 0; i < result->ne[0]/* cols */; i++) {
            printf(" %.2f", result_data[j * result->ne[0] + i]);
        }
    }
    printf(" ]\n");
   
    // 6. Free memory and exit
    ggml_free(ctx);
    return 0;
}

