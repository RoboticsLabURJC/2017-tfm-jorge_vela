/** ------------------------------------------------------------------------
 *
 *  @brief Channel feature extractors for LUV color space.
 *  @author Jorge Vela
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2020/10/31
 *
 *  ------------------------------------------------------------------------ */

#ifndef CHANNELS_EXTRACTOR_LUV_OPENCV
#define CHANNELS_EXTRACTOR_LUV_OPENCV

#include <channels/ChannelsExtractorLUV.h>
//#include <opencv2/opencv.hpp>
//#include <vector>

class ChannelsExtractorLUVOpenCV: public ChannelsExtractorLUV
{
public:

  ChannelsExtractorLUVOpenCV
    (
    bool smooth = true,
    int smooth_kernel_size = 1
    );

  ~ChannelsExtractorLUVOpenCV
    () {}

  virtual std::vector<cv::Mat> extractFeatures
    (
      cv::Mat img
    );

private:

  float m_smooth;
  int m_smooth_kernel_size;

  cv::Mat smoothImage
    (
    cv::Mat imgLUV
    );
};

#endif
