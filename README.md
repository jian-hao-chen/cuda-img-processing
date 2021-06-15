# cuda-img-processing

臺灣師範大學 109 學年度第 2 學期, 電機工程學系平行運算課程期末專題

本專題使用 CUDA 函式庫透過 NVIDIA 的 GPU 對一些影像處理的方法做加速



## Requirements

### Hardware & OS

- Linux OS
- NVIDIA GPU

### Libraries

- OpenCV 4.5.0
- CUDA 11.0
- CMake >= 3.9



## How to Build

1. 本專題有使用 OpenCV 對 CUDA 的接口, 所以需要**先安裝 CUDA**

2. 確定 CUDA 安裝成功後, 從原始碼用 CMake 編譯 OpenCV

   ```shell
   $ ./install_opencv.sh
   ```

3. 之後就可以使用 CMake 編譯本專案, 由於有使用到 Linux 的 `getopt` 函式庫, 所以只有 Linux 作業系統下可以正常編譯

   ```shell
   $ ./build.sh
   ```

   

