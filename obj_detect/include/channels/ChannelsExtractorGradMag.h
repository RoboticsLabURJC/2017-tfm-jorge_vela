
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

#define USE_OPENCV_IMPLEMENTATION

class GradMagExtractor
{
public:
  GradMagExtractor
    (
      int normRad = 0,
      float normConst = 0.005
    )
  {
    m_normRad = normRad;
    m_normConst = normConst;
  };
    
  std::vector<cv::Mat> extractFeatures
    (
      cv::Mat img 
    );

private:

#ifdef USE_OPENCV_IMPLEMENTATION
  std::vector<cv::Mat> extractFeaturesOpenCV
    (
      cv::Mat img
    );
#else
  std::vector<cv::Mat> extractFeaturesPDollar
    (
      cv::Mat img
    );
#endif

  int m_normRad;
  float m_normConst;
};

#endif
