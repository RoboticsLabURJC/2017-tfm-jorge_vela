#include <opencv2/core/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/opencv.hpp>
#include <chrono>

using namespace cv;

int main(int argc, char** argv)
{
  cv::ocl::Context context;
  if (!context.create(cv::ocl::Device::TYPE_GPU))
  {
    std::cout << "Failed creating the context..." << std::endl;
  }
  cv::ocl::Device(context.device(0));
  cv::ocl::setUseOpenCL(true);

  Mat img1 = imread("../tests/images/coches10.jpg", IMREAD_COLOR);
  Mat gray_cpu;
  resize(img1, img1, Size(0,0), 5, 5, INTER_LINEAR);

  auto start = std::chrono::system_clock::now();
  UMat img, gray;
  img1.copyTo(img); // include time to upload image from CPU -> GPU

  cvtColor(img, gray, cv::COLOR_BGR2GRAY);
  cv::GaussianBlur(gray, gray,Size(7, 7), 1.5);
  cv::Canny(gray, gray, 0, 50);

  gray.copyTo(gray_cpu); // include time to download result from GPU -> CPU

  auto end = std::chrono::system_clock::now();
  std::chrono::duration<float,std::milli> duration = end - start;
  std::cout << duration.count() << "ms" << std::endl;

  return 0;
}


