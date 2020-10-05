
/** ------------------------------------------------------------------------
 *
 *  @brief Channel feature extractors for magnitude and orient gradients.
 *  @author Jorge Vela
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2019/07/08
 *
 *  ------------------------------------------------------------------------ */
//ChanExtractorMex.h

#ifndef CHANNELS_GRADMAG
#define CHANNELS_GRADMAG

#include <opencv2/opencv.hpp>
#include <vector>

class GradMagExtractor
{
public:
  GradMagExtractor
    (
      int normRad = 0,
      float normConst = 0.005,
      bool use_opencv_impl = true
    )
  {
    m_normRad = normRad;
    m_normConst = normConst;
    m_use_opencv_impl = true;
  };
    
  std::vector<cv::Mat> extractFeatures
    (
      cv::Mat img 
    );

private:

  std::vector<cv::Mat> extractFeaturesOpenCV
    (
      cv::Mat img
    );

  std::vector<cv::Mat> extractFeaturesPDollar
    (
      cv::Mat img
    );

  int m_normRad;
  float m_normConst;
  bool m_use_opencv_impl;
};

#endif
