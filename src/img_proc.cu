#include "img_proc.cuh"

__global__ void mirror_kernel(cv::cuda::PtrStepSz<uchar3> src, cv::cuda::PtrStepSz<uchar3> dst, int h, int w)
{
        unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
        unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
        if (x < src.cols && y < src.rows) {
                dst(y, x) = src(y, w - x - 1);
        }
}

void mirror_cu(cv::cuda::GpuMat src, cv::cuda::GpuMat dst, int h, int w)
{
        assert(src.cols == w && src.rows == h);
        dim3 block(32, 32);
        unsigned int num_block_x = (w + block.x - 1) / block.x;
        unsigned int num_block_y = (h + block.y - 1) / block.y;
        dim3 grid(num_block_x, num_block_y);
        mirror_kernel <<<grid, block>>> (src, dst, h, w);
}