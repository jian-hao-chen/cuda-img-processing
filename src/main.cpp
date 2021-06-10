#include <stdlib.h>

#include <iostream>
#include <opencv2/opencv.hpp>

void mirror_cu(cv::cuda::GpuMat src, cv::cuda::GpuMat dst, int h, int w);

int main(int argc, char* argv[])
{
        /* Checks arg format. */
        if (argc < 2) {
                std::cout << "[ERROR] Please enter the path of video." << std::endl;
                return EXIT_FAILURE;
        }

        /* Parses args. */
        std::cout << "[INFO] Input file: " << argv[1] << std::endl;
        for (size_t i = 2; i < argc; i++) {
                std::cout << "[INFO] arg" << i << ": " << argv[i] << std::endl;
        }

        /* Opens video. */
        cv::VideoCapture cap(argv[1]);
        if (!cap.isOpened()) {
                std::cout << "[ERROR] " << argv[1] << " can't be opened!" << std::endl;
        }

        cv::String win_name = cv::String(argv[0]);
        cv::namedWindow(win_name);

        int h = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        int w = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        /* Reads frames. */
        while (true) {
                cv::Mat frame;
                cap >> frame;
                if (frame.empty()) {
                        break;
                }
                cv::Mat dst(h, w, CV_8UC3, cv::Scalar(0, 0, 0));  // Mat on CPU memory.
                cv::cuda::GpuMat cu_dst(h, w, CV_8UC3, cv::Scalar(0, 0, 0));  // Mat on GPU memory.
                cv::cuda::GpuMat cu_src;
                cu_src.upload(frame);
                mirror_cu(cu_src, cu_dst, h, w);
                cu_dst.download(dst);

                cv::imshow(win_name, dst);

                /* Presses ESC to exit. */
                if (cv::waitKey(10) == 27) {
                        break;
                }
        }
        cap.release();
        cv::destroyAllWindows();

        return EXIT_SUCCESS;
}