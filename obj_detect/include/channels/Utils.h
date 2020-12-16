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

cv::Mat ImgResample
  (
    cv::Mat src,
    int width,
    int height,
    std::string method = "antialiasing",
    float norm = 1
  );

cv::Mat convTri
  (
  cv::Mat input_image,
  int kernel_size
  );

cv::UMat convTri
  (
  cv::UMat input_image,
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
