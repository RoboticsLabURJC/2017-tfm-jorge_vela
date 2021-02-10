/** ------------------------------------------------------------------------
 *
 *  @brief Channel Utils.
 *  @author Jorge Vela
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2020/17/02
 *
 *  ------------------------------------------------------------------------ */

#ifndef IMAGE_UTILS
#define IMAGE_UTILS

#include <opencv2/opencv.hpp>
#include <vector>

void
ImgResample
  (
  cv::Mat src,
  cv::Mat& dst,
  int width,
  int height,
  std::string method = "antialiasing",
  float norm = 1
  );

void
ImgResample
  (
  cv::UMat src,
  cv::UMat& dst,
  int width,
  int height,
  std::string method = "antialiasing",
  float norm = 1
  );

void
convTri
  (
  cv::Mat input_image,
  cv::Mat& dst,
  int kernel_size
  );

void
convTri
  (
  cv::UMat input_image,
  cv::UMat& dst,
  int kernel_size
  );

float
readScalarFromFileNode
  (
  cv::FileNode fn
  );

cv::Mat
readMatrixFromFileNode
  (
  cv::FileNode fn
  );

cv::Mat
readMatrixFromFileNodeWrongBufferMatlab
  (
  cv::FileNode fn
  );

std::vector<int>
create_random_indices
  (
  int n
  );

#endif
