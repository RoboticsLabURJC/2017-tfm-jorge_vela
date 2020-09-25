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
    int norm = 1
  );

std::vector<cv::Mat> channelsCompute
  (
  cv::Mat src,
  std::string colorSpace,
  int shrink
  );


cv::Mat convTri
  (
  cv::Mat input_image,
  int kernel_size
  );

#endif
