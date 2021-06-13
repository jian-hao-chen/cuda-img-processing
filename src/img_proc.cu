#include "img_proc.cuh"

__global__ void mirror_kernel(cv::cuda::PtrStepSz<uchar3> src,
                              cv::cuda::PtrStepSz<uchar3> dst, int h, int w)
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
        CHECK_ERROR(cudaDeviceSynchronize());
}

__global__ void gamma_correct_kernel(cv::cuda::PtrStepSz<uchar3> src,
                                     cv::cuda::PtrStepSz<uchar3> dst,
                                     int h, int w, float gamma)
{
        unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
        unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
        if (x < src.cols && y < src.rows) {
                float b = src(y, x).x / 255.0f;
                float g = src(y, x).y / 255.0f;
                float r = src(y, x).z / 255.0f;
                dst(y, x).x = (unsigned char)(255 * powf(b, gamma));
                dst(y, x).y = (unsigned char)(255 * powf(g, gamma));
                dst(y, x).z = (unsigned char)(255 * powf(r, gamma));
        }
        
}

void gamma_correct_cu(cv::cuda::GpuMat src, cv::cuda::GpuMat dst, int h, int w,
                      float gamma)
{
        assert(src.cols == w && src.rows == h);
        dim3 block(32, 32);
        unsigned int num_block_x = (w + block.x - 1) / block.x;
        unsigned int num_block_y = (h + block.y - 1) / block.y;
        dim3 grid(num_block_x, num_block_y);
        gamma_correct_kernel <<<grid, block>>> (src, dst, h, w, gamma);
        CHECK_ERROR(cudaDeviceSynchronize());
}

__global__ void gaussian_kernel(cv::cuda::PtrStepSz<uchar3> src,
                                cv::cuda::PtrStepSz<uchar3> dst,
                                int h, int w)
{
        float filter[3][3] = {{0.0625f, 0.125f, 0.0625f},
                              { 0.125f,  0.25f,  0.125f},
                              {0.0625f, 0.125f, 0.0625f}};
        unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
        unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
        if (0 < x && x < src.cols -1 && 0 < y && y < src.rows - 1) {
                float b = 0;
                float g = 0;
                float r = 0;
                for (size_t i = 0; i < 3; i++) {
                        for (size_t j = 0; j < 3; j++) {
                                b += src(y - 1 + j, x - 1 + i).x * filter[i][j];
                                g += src(y - 1 + j, x - 1 + i).y * filter[i][j];
                                r += src(y - 1 + j, x - 1 + i).z * filter[i][j];
                        }
                }
                dst(y, x).x = (unsigned char)b;
                dst(y, x).y = (unsigned char)g;
                dst(y, x).z = (unsigned char)r;
        }
}

void gaussian_cu(cv::cuda::GpuMat src, cv::cuda::GpuMat dst, int h, int w)
{
        assert(src.cols == w && src.rows == h);
        src.copyTo(dst);
        dim3 block(32, 32);
        unsigned int num_block_x = (w + block.x - 1) / block.x;
        unsigned int num_block_y = (h + block.y - 1) / block.y;
        dim3 grid(num_block_x, num_block_y);
        gaussian_kernel <<<grid, block>>> (src, dst, h, w);
        CHECK_ERROR(cudaDeviceSynchronize());
}