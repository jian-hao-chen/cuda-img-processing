#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/>

#define CHECK_ERROR(call) {\
    const cudaError_t err = call;\
    if (err != cudaSuccess)\
    {\
        printf("Error:%s,%d,",__FILE__,__LINE__);\
        printf("code:%d,reason:%s\n",err,cudaGetErrorString(err));\
        exit(1);\
    }\
}

__global__ void mirror_kernel(cv::cuda::PtrStepSz<uchar3> src, cv::cuda::PtrStepSz<uchar3> dst, int h, int w)
{
        unsigned int x;
        unsigned int y;
}