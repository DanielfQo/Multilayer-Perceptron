cmake_minimum_required(VERSION 3.18)
project(mlp_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Habilitar separable compilation para poder llamar kernels en diferentes archivos .cu
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

add_executable(mlp_cuda
    main.cpp
    cuda_kernels/cuda_forward.cu
    cuda_kernels/cuda_backward.cu
    cuda_kernels/cuda_update.cu
    utils/mnist.cpp
    utils/cargar_simpsons.cpp
)

target_include_directories(mlp_cuda PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/mlp
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_kernels
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
)

# Enlazar la librería de CUDA para que C++ encuentre los includes
find_package(CUDAToolkit REQUIRED)
target_link_libraries(mlp_cuda PRIVATE CUDA::cudart)

if(MSVC)
    target_compile_options(mlp_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--std=c++17>)
endif()
