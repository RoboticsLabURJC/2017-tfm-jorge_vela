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
      std::string colorSpace,
      int shrink
    );


  cv::Mat convTri
    (
      cv::Mat input_image,
      int kernel_size
    );

  /*std::vector<cv::Mat> chnsPyramids
    (
      cv::Mat img,
      int nOctUp =0,
      int nPerOct = 8,
      int nApprox = 7,
      int shrink = 4,
      std::vector<int> minDsA= {16,16}
    );*/


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