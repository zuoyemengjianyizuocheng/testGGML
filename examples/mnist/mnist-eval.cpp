#include "ggml.h"
#include "ggml-opt.h"

#include "mnist-common.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <thread>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

int main(int argc, char** argv) {
    srand(time(NULL));
    ggml_time_init();

    if (argc != 4 && argc != 5) {
        fprintf(stderr, "Usage: %s mnist-fc-f32.gguf data/MNIST/raw/t10k-images-idx3-ubyte data/MNIST/raw/t10k-labels-idx1-ubyte [CPU/CUDA0]\n", argv[0]);
        //exit(1);
    }
    char str1[100] = "C:/Users/cjj/Documents/testGGML/examples/mnist/mnist-fc-f32.gguf";
    char str2[100] = "C:/Users/cjj/Documents/testGGML/examples/mnist/data/MNIST/raw/t10k-images-idx3-ubyte";
    char str3[100] = "C:/Users/cjj/Documents/testGGML/examples/mnist/data/MNIST/raw/t10k-labels-idx1-ubyte";
    argv[1] = str1;
    argv[2] = str2;
    argv[3] = str3;

    //MNIST_NINPUT，图片size28*28
    //MNIST_NCLASSES，10类
    //MNIST_NTEST 10000张
    //MNIST_NBATCH_PHYSICAL 批处理的单批batch大小
    //申请空间
    ggml_opt_dataset_t dataset = ggml_opt_dataset_init(MNIST_NINPUT, MNIST_NCLASSES, MNIST_NTEST, MNIST_NBATCH_PHYSICAL);

    //读取数据
    if (!mnist_image_load(argv[2], dataset)) {
        return 1;
    }
    if (!mnist_label_load(argv[3], dataset)) {
        return 1;
    }

    // 随机展示样例图像
    const int iex = rand() % MNIST_NTEST;
    mnist_image_print(stdout, dataset, iex);

    // 模型初始化
    const std::string backend = argc >= 5 ? argv[4] : "";  // 计算后端

    // 计算时间
    const int64_t t_start_us = ggml_time_us();
    // 模型初始化--模型、后端、逻辑批处理大小决定有多少数据点用于梯度更新、物理批处理大小决定并行处理多少数据点
    mnist_model model = mnist_model_init_from_file(argv[1], backend, MNIST_NBATCH_LOGICAL, MNIST_NBATCH_PHYSICAL);
    // 构建计算图
    mnist_model_build(model);
    const int64_t t_load_us = ggml_time_us() - t_start_us;
    fprintf(stdout, "%s: loaded model in %.2lf ms\n", __func__, t_load_us / 1000.0);

    // 模型评估
    ggml_opt_result_t result_eval = mnist_model_eval(model, dataset);

    // 结果解析--预测结果
    std::vector<int32_t> pred(MNIST_NTEST);
    ggml_opt_result_pred(result_eval, pred.data());
    fprintf(stdout, "%s: predicted digit is %d\n", __func__, pred[iex]);

    // 计算损失和准确率
    double loss;
    double loss_unc;
    ggml_opt_result_loss(result_eval, &loss, &loss_unc);
    fprintf(stdout, "%s: test_loss=%.6lf+-%.6lf\n", __func__, loss, loss_unc);

    double accuracy;
    double accuracy_unc;
    ggml_opt_result_accuracy(result_eval, &accuracy, &accuracy_unc);
    fprintf(stdout, "%s: test_acc=%.2lf+-%.2lf%%\n", __func__, 100.0 * accuracy, 100.0 * accuracy_unc);

    // 释放资源
    ggml_opt_result_free(result_eval);

    return 0;
}
