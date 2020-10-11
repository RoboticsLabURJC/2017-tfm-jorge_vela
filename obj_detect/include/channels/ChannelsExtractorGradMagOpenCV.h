/** ------------------------------------------------------------------------
 *
 *  @brief Channel feature extractors for magnitude and orient gradients.
 *  @author Jorge Vela
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2019/07/08
 *
 *  ------------------------------------------------------------------------ */

#ifndef CHANNELS_EXTRACTOR_GRAD_MAG_OPENCV
#define CHANNELS_EXTRACTOR_GRAD_MAG_OPENCV

#include <channels/ChannelsExtractorGradMag.h>
#include <opencv2/opencv.hpp>
#include <vector>

class ChannelsExtractorGradMagOpenCV: public ChannelsExtractorGradMag
{
public:
  ChannelsExtractorGradMagOpenCV
    (
      int normRad = 0,
      float normConst = 0.005,
      bool use_opencv_impl = true
    ): ChannelsExtractorGradMag(normRad, normConst, use_opencv_impl)
  {};

  virtual ~ChannelsExtractorGradMagOpenCV() {};
    
  virtual std::vector<cv::Mat> extractFeatures
    (
      cv::Mat img 
    );
};

#endif
