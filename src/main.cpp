#include <stdlib.h>

#include <iostream>
#include <opencv2/opencv.hpp>

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

        /* Reads frames. */
        while (true) {
                cv::Mat frame;
                cap >> frame;
                if (frame.empty()) {
                        break;
                }
                cv::imshow(win_name, frame);

                /* Presses ESC to exit. */
                if (cv::waitKey(10) == 27) {
                        break;
                }
        }
        cap.release();
        cv::destroyAllWindows();

        return EXIT_SUCCESS;
}