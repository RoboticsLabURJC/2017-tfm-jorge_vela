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


#include <opencv/cv.hpp>
#include <vector>

class Utils{
public:
  cv::Mat ImgResample
    (
      cv::Mat src, 
      int width,
      int height,
      int norm = 1
    );


  struct productChnsCompute 
    {
      cv::Mat image;
      float* M;
      float* O;
      float* H;
    } ;


  std::vector<cv::Mat> channelsCompute
    (
      cv::Mat src,
      int shrink
    );


  cv::Mat convTri
    (
      cv::Mat input_image,
      int kernel_size
    );

  void chnsPyramids
    (
      cv::Mat img
    );
  std::vector<float> getScales
    (
      int nPerOct,
      int nOctUp,
      int minDs[],
      int shrink,
      int sz[]
    );
};
#endif