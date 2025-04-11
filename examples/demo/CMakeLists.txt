#
set(TEST_TARGET demo)
add_executable(${TEST_TARGET} demo )
#find_package(ggml REQUIRED)
#target_link_libraries(${TEST_TARGET}  PUBLIC ggml::large_project_target)
target_link_libraries(${TEST_TARGET} PRIVATE  ggml)


