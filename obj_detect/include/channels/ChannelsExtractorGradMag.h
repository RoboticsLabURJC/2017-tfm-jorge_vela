
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
  int m_normRad;
  float m_normConst;

  float*
  allocW
    (
      int size,
      int sf,
      int misalign
    );

  std::vector<cv::Mat>
  gradM
    (
      cv::Mat image,
      float* M,
      float* O
    );

  void
  gradMagNorm
    (
      float *M,
      float *S,
      int h,
      int w,
      float norm
    );
};

#endif
