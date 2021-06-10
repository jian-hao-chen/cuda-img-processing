cmake_minimum_required(VERSION 3.9)
project(cuda-img-proc
	LANGUAGES CXX CUDA
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_library(img_proc img_proc.cu img_proc.cuh)
target_compile_features(img_proc PUBLIC cxx_std_11)

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} img_proc)