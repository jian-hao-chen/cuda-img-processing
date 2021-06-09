cmake_minimum_required(VERSION 3.9)
project(cuda-img-proc
	LANGUAGES CXX CUDA
)

set(CMAKE_BINARY_DIR ./build)
set(CMAKE_SOURCE_DIR .)

add_subdirectory(src ${CMAKE_BINARY_DIR})