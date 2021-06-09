set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories(../opencv/include)

find_package(CUDAToolkit REQUIRED)
add_library(img_proc img_proc.cu)
target_compile_features(img_proc PUBLIC cxx_std_11)
target_link_libraries(img_proc CUDA::cudart)

add_executable(${PROJECT_NAME} main.cpp)
if(WIN32)
	target_link_libraries(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/pre-built/vc16/lib/opencv_world452d.lib)
endif()
#target_include_directories(${PROJECT_NAME} PRIVATE ../include)