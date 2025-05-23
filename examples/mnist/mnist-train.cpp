#include "ggml-opt.h"
#include "mnist-common.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

int main(int argc, char ** argv) {
    argc = 5;
    if (argc != 5 && argc != 6) {
        fprintf(stderr, "Usage: %s mnist-fc mnist-fc-f32.gguf data/MNIST/raw/train-images-idx3-ubyte data/MNIST/raw/train-labels-idx1-ubyte [CPU/CUDA0]\n", argv[0]);
        exit(0);
    }
    char str1[100] = "mnist-fc";
    //char str0[100] = "C:/Users/cjj/Documents/testGGML/examples/mnist/data/MNIST/raw/mnist-fc-f32.gguf";
    char str2[100] = "C:/Users/cjj/Documents/testGGML/examples/mnist/data/MNIST/raw/train-images-idx3-ubyte";
    char str3[100] = "C:/Users/cjj/Documents/testGGML/examples/mnist/data/MNIST/raw/train-labels-idx1-ubyte";
    /*argv[1] = str0;
    argv[2] = str1;*/
    argv[3] = str2;
    argv[4] = str3;
    // The MNIST model is so small that the overhead from data shuffling is non-negligible, especially with CUDA.
    // With a shard size of 10 this overhead is greatly reduced at the cost of less shuffling (does not seem to have a significant impact).
    // A batch of 500 images then consists of 50 random shards of size 10 instead of 500 random shards of size 1.
    ggml_opt_dataset_t dataset = ggml_opt_dataset_init(MNIST_NINPUT, MNIST_NCLASSES, MNIST_NTRAIN, /*ndata_shard =*/ 10); //训练数据空间申请，并初始化节点

    if (!mnist_image_load(argv[3], dataset)) { //读取数据
        return 1;
    }
    if (!mnist_label_load(argv[4], dataset)) {
        return 1;
    }

    mnist_model model = mnist_model_init_random(argv[1], argc >= 6 ? argv[5] : "", MNIST_NBATCH_LOGICAL, MNIST_NBATCH_PHYSICAL);    //模型初始化，包括网络结构，各类变量，包括分类结果等，都初始化空间

    mnist_model_build(model);       //模型各层之间的连接关系，参数层如何优化，输出结果怎么出来等

    mnist_model_train(model, dataset, /*nepoch =*/ 30, /*val_split =*/ 0.05f);      //模型训练，包括参数优化，损失计算，循环30次，验证集比例5%

    mnist_model_save(model, argv[2]);       //存储模型
}
