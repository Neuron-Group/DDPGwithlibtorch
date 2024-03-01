
aux_source_directory (. SRC_LIST)
include_directories (../include)

# 如果使用 torch 内置的 libtorch
# set(CMAKE_PREFIX_PATH /home/neuron/anaconda3/envs/PyTorch/lib/python3.8/site-packages/torch/share/cmake)
execute_process(COMMAND python -c "import torch;print(torch.utils.cmake_prefix_path)"
                OUTPUT_VARIABLE TORCH_PATH
                OUTPUT_STRIP_TRAILING_WHITESPACE)
set(CMAKE_PREFIX_PATH ${TORCH_PATH})

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FALGS}")

add_compile_options(-fpermissive)

add_executable(${PROJECT_NAME} ${SRC_LIST})
set (EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)


# 添加链接库
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)


