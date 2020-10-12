/** ------------------------------------------------------------------------
 *
 *  @brief DetectionRectangle.
 *  @author Jorge Vela
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2020/09/29
 *
 *  ------------------------------------------------------------------------ */
#ifndef CLASSIFIER_CONFIG_HPP
#define CLASSIFIER_CONFIG_HPP

#include <opencv2/opencv.hpp>


  struct channelsLUV{
    int smooth_kernel_size;
    int smooth;
  };

  struct gradientMag{
    int normRad;
    float normConst;
  };

  struct gradientHist{
    int binSize;
    int nOrients;
    int softBin;
    int full;
  };
/** ------------------------------------------------------------------------
 *
 *  @brief Struct whit the classifier values to extract data LUV, MAG, HIST.
 *
 *  ------------------------------------------------------------------------ */
struct ClassifierConfig
{

  cv::Size padding;
  int nOctUp;
  int nPerOct;
  int nApprox;
  int shrink;
  //int stride;

  channelsLUV luv;
  gradientMag gradMag;
  gradientHist gradHist;

  std::vector<float> lambdas;
  cv::Size minDs;

  cv::Size modelDsPad;
  cv::Size modelDs;
  float cascThr;

  int stride;

};
#endif // DETECTION_HPP
