/** ------------------------------------------------------------------------
 *
 *  @brief Channel feature extractors for magnitude and orient gradients.
 *  @author Jorge Vela
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2019/07/08
 *
 *  ------------------------------------------------------------------------ */

#ifndef CHANNELS_EXTRACTOR_GRAD_MAG
#define CHANNELS_EXTRACTOR_GRAD_MAG

#include <opencv2/opencv.hpp>
#include <vector>

class ChannelsExtractorGradMag
{
public:
  ChannelsExtractorGradMag
    (
      int normRad = 0,
      float normConst = 0.005,
      bool use_opencv_impl = true
    )
  {
    m_normRad = normRad;
    m_normConst = normConst;
    m_use_opencv_impl = use_opencv_impl;
  };

  virtual ~ChannelsExtractorGradMag
    () {}
    
  virtual std::vector<cv::Mat> extractFeatures
    (
      cv::Mat img 
    ) = 0;

protected:
  int m_normRad;
  float m_normConst;
  bool m_use_opencv_impl;
};

#endif
