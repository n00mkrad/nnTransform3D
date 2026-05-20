cmake_minimum_required(VERSION 3.18)
project(nnTransform3D LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 75 86 89 120)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)


set(ONNXRUNTIME_DIR "${CMAKE_SOURCE_DIR}/onnxruntime-win-x64-gpu" CACHE PATH "Path to ONNX Runtime")

include_directories(${ONNXRUNTIME_DIR}/include)

if(WIN32)
    set(ONNXRUNTIME_LIB "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")
else()
    set(ONNXRUNTIME_LIB "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
endif()

add_executable(nnTransform3D
    main.cu
    y4m_ntsc.cpp
)

target_link_libraries(nnTransform3D PRIVATE
    cufft
    ${ONNXRUNTIME_LIB}
)
