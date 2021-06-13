#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

#include <opencv2/opencv.hpp>

int VIDEO_CODEC = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
char *SRC_PATH;
char *DST_PATH;

bool IS_MIRROR = false;
bool IS_GAMMA_CORRECT = false;
bool IS_GAUSSIAN = false;
float GAMMA;

void parse_args(int argc, char *argv[]);
void mirror_cu(cv::cuda::GpuMat src, cv::cuda::GpuMat dst, int h, int w);
void gamma_correct_cu(cv::cuda::GpuMat src, cv::cuda::GpuMat dst, int h, int w,
                      float gamma);
void gaussian_cu(cv::cuda::GpuMat src, cv::cuda::GpuMat dst, int h, int w);

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

        /* Reads frames. */
        while (true) {
                cv::Mat frame;
                cap >> frame;
                if (frame.empty()) {
                        break;
                }
                /* Mat on RAM for CPU. */
                cv::Mat dst(h, w, CV_8UC3, cv::Scalar(0, 0, 0));
                /* Mat on VRAM for GPU. */
                cv::cuda::GpuMat cu_dst(h, w, CV_8UC3, cv::Scalar(0, 0, 0));
                cv::cuda::GpuMat cu_src;
                cu_src.upload(frame);
                
                if (IS_MIRROR) {
                        mirror_cu(cu_src, cu_dst, h, w);
                        cu_dst.copyTo(cu_src);
                }
                if (IS_GAMMA_CORRECT) {
                        gamma_correct_cu(cu_src, cu_dst, h, w, GAMMA);
                }

                cu_dst.download(dst);
                video.write(dst);
        }
        
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
                cmd_opt = getopt(argc, argv, "mg:");

                /* All args were parsed. */
                if (cmd_opt == -1) {
                        break;
                }

                switch (cmd_opt) {
                case 'd':
                        IS_GAUSSIAN = true;
                        printf("[INFO] -d Enable denoise.");
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