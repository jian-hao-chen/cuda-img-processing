#ifndef IMG_PROC_H
#define IMG_PROC_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "opencv2/cudawarping.hpp"

#define CHECK_ERROR(call) {\
    const cudaError_t err = call;\
    if (err != cudaSuccess)\
    {\
        printf("Error:%s,%d,",__FILE__,__LINE__);\
        printf("code:%d,reason:%s\n",err,cudaGetErrorString(err));\
        exit(1);\
    }\
}

void mirror_cu(cv::cuda::GpuMat src, cv::cuda::GpuMat dst, int h, int w);

#endif // !IMG_PROC_H