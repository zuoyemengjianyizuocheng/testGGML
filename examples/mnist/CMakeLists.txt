#
# mnist-common
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(TEST_TARGET mnist-common)
add_library(${TEST_TARGET} STATIC mnist-common.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common)

#
# mnist-eval

set(TEST_TARGET mnist-eval)
add_executable(${TEST_TARGET} mnist-eval.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common mnist-common)

#
# mnist-train

set(TEST_TARGET mnist-train)
add_executable(${TEST_TARGET} mnist-train.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common mnist-common)
