#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include <opencv2/opencv.hpp>

int VIDEO_CODEC = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
char *SRC_PATH;
char *DST_PATH;

bool IS_CPU = false;
bool IS_MIRROR = false;
bool IS_GAMMA_CORRECT = false;
bool IS_DENOISE = false;
float GAMMA;

void parse_args(int argc, char *argv[]);
void mirror_cu(cv::cuda::GpuMat src, cv::cuda::GpuMat dst, int h, int w);
void gamma_correct_cu(cv::cuda::GpuMat src, cv::cuda::GpuMat dst, int h, int w,
                      float gamma);
void gaussian_cu(cv::cuda::GpuMat src, cv::cuda::GpuMat dst, int h, int w);
void median_cu(cv::cuda::GpuMat src, cv::cuda::GpuMat dst, int h, int w);
void mirror(cv::Mat &src, cv::Mat &dst, int h, int w);
void gamma_correct(cv::Mat &src, cv::Mat &dst, int h, int w, float gamma);
void gaussian(cv::Mat &src, cv::Mat &dst, int h, int w);

int main(int argc, char *argv[])
{
        parse_args(argc, argv);

        /* Opens video. */
        cv::VideoCapture cap(SRC_PATH);
        if (!cap.isOpened()) {
                printf("[ ERR] %s can't be opened!", SRC_PATH);
        }
        int h = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        int w = cap.get(cv::CAP_PROP_FRAME_WIDTH);

        /* Creats video writer. */
        cv::VideoWriter video(DST_PATH, VIDEO_CODEC, 25, cv::Size(w, h));

        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        /* Reads frames. */
        while (true) {
                cv::Mat frame;
                cap >> frame;
                if (frame.empty()) {
                        break;
                }
                /* Mat on RAM for CPU. */
                cv::Mat dst(h, w, CV_8UC3, cv::Scalar(0, 0, 0));
                
                if (IS_CPU) {
                        cv::Mat src(frame);
                        if (IS_MIRROR) {
                                mirror(src, dst, h, w);
                                dst.copyTo(src);
                        }
                        if (IS_DENOISE) {
                                gaussian(src, dst, h, w);
                                dst.copyTo(src);
                        }
                        if (IS_GAMMA_CORRECT) {
                                gamma_correct(src, dst, h, w, GAMMA);
                        }
                }
                else {
                        /* Mat on VRAM for GPU. */
                        cv::cuda::GpuMat cu_dst(h, w, CV_8UC3, cv::Scalar(0, 0, 0));
                        cv::cuda::GpuMat cu_src;
                        cu_src.upload(frame);
                        if (IS_MIRROR) {
                                mirror_cu(cu_src, cu_dst, h, w);
                                cu_dst.copyTo(cu_src);
                        }
                        if (IS_DENOISE) {
                                gaussian_cu(cu_src, cu_dst, h, w);
                                cu_dst.copyTo(cu_src);
                        }
                        if (IS_GAMMA_CORRECT) {
                                gamma_correct_cu(cu_src, cu_dst, h, w, GAMMA);
                        }
                        cu_dst.download(dst);
                }
                video.write(dst);
        }

        clock_gettime(CLOCK_MONOTONIC, &end);
        printf("[INFO] Elapsed time: %ld s.\n", end.tv_sec - start.tv_sec);
        
        /* All Done. */
        cap.release();
        video.release();

        return EXIT_SUCCESS;
}

void parse_args(int argc, char *argv[])
{
        if (argc < 3) {
                printf("Usage: cuda-img-proc [-d] <src_path> <dst_path>\n");
                exit(EXIT_FAILURE);
        }

        int cmd_opt = 0;
        while (true) {
                cmd_opt = getopt(argc, argv, "cdmg:");

                /* All args were parsed. */
                if (cmd_opt == -1) {
                        break;
                }

                switch (cmd_opt) {
                case 'c':
                        IS_CPU = true;
                        printf("[INFO] -c CPU mode.\n");
                        break;
                case 'd':
                        IS_DENOISE = true;
                        printf("[INFO] -d Enable denoise.\n");
                        break;
                case 'g':
                        IS_GAMMA_CORRECT = true;
                        GAMMA = atof(optarg);
                        printf("[INFO] -g Enable gamma correction, ");
                        printf("GAMMA=%.2f\n", GAMMA);
                        break;
                case 'm':
                        IS_MIRROR = true;
                        printf("[INFO] -m Enable mirror.\n");
                        break;
                case '?':
                        fprintf(stderr, "[WARN] Unknown argument: -%c\n",
                                optopt);
                        break;
                default:
                        break;
                }        
        }

        /* Gets the remaining args. */
        if (argc > optind) {
                SRC_PATH = argv[optind];
                DST_PATH = argv[optind + 1];
        }
}

void mirror(cv::Mat &src, cv::Mat &dst, int h, int w)
{
        for (size_t y = 0; y < h; y++) {
                for (size_t x = 0; x < w; x++) {
                        dst.at<cv::Vec3b>(y, x) = src.at<cv::Vec3b>(y, w - x - 1);
                }
        }  
}

void gamma_correct(cv::Mat &src, cv::Mat &dst, int h, int w, float gamma)
{
        for (size_t y = 0; y < h; y++) {
                for (size_t x = 0; x < w; x++) {
                        cv::Vec3f tmp = src.at<cv::Vec3b>(y, x);
                        tmp[0] = 255 * std::pow(tmp[0] / 255.0f, gamma);
                        tmp[1] = 255 * std::pow(tmp[1] / 255.0f, gamma);
                        tmp[2] = 255 * std::pow(tmp[2] / 255.0f, gamma);
                        dst.at<cv::Vec3b>(y, x)[0] = (unsigned char)tmp[0];
                        dst.at<cv::Vec3b>(y, x)[1] = (unsigned char)tmp[1];
                        dst.at<cv::Vec3b>(y, x)[2] = (unsigned char)tmp[2];
                }
        }  
}

void gaussian(cv::Mat &src, cv::Mat &dst, int h, int w)
{
        float filter[5][5] = {{ 0.003765, 0.015019, 0.023792, 0.015019, 0.003765 },
                              { 0.015019, 0.059912, 0.094907, 0.059912, 0.015019 },
                              { 0.023792, 0.094907, 0.150342, 0.094907, 0.023792 },
                              { 0.015019, 0.059912, 0.094907, 0.059912, 0.015019 },
                              { 0.003765, 0.015019, 0.023792, 0.015019, 0.003765 }};
        for (size_t y = 2; y < h - 2; y++) {
                for (size_t x = 2; x < w - 2; x++) {
                        cv::Vec3f tmp(0, 0, 0);
                        for (size_t i = 0; i < 5; i++) {
                                for (size_t j = 0; j < 5; j++) {
                                        cv::Vec3b p = src.at<cv::Vec3b>(y - 2 + i, x - 2 + j);
                                        tmp[0] += p[0] * filter[i][j];
                                        tmp[1] += p[1] * filter[i][j];
                                        tmp[2] += p[2] * filter[i][j];
                                }
                                
                        }
                        dst.at<cv::Vec3b>(y, x)[0] = (unsigned char)tmp[0];
                        dst.at<cv::Vec3b>(y, x)[1] = (unsigned char)tmp[1];
                        dst.at<cv::Vec3b>(y, x)[2] = (unsigned char)tmp[2];
                }
        }  
}