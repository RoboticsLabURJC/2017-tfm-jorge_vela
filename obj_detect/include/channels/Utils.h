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


cv::Mat ImgResample
  (
    cv::Mat src, 
    int width,
    int height,
    int nChannels
  );


struct productChnsCompute 
  {
    cv::Mat image;
    float* M;
    float* O;
    float* H;
  } ;


productChnsCompute channelsCompute
  (
    cv::Mat src,
    int shrink
  );


cv::Mat convTri
  (
    cv::Mat input_image,
    int kernel_size
  );

void getScales
  (
    int nPerOct,
    int nOctUp,
    int minDs[],
    int shrink,
    int sz[]
  );
#endif