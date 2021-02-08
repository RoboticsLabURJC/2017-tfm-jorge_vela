/** ------------------------------------------------------------------------
 *
 *  @brief Channel feature extractors for LUV color space (using T-API)
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2021/01/29
 *
 *  ------------------------------------------------------------------------ */

#ifndef CHANNELS_EXTRACTOR_LUV_OPENCL
#define CHANNELS_EXTRACTOR_LUV_OPENCL

#include <channels/ChannelsExtractorLUV.h>
//#include <opencv2/opencv.hpp>
//#include <vector>

class ChannelsExtractorLUVOpenCL: public ChannelsExtractorLUV
{
public:

  ChannelsExtractorLUVOpenCL
    (
    bool smooth = true,
    int smooth_kernel_size = 1
    );

  ~ChannelsExtractorLUVOpenCL
    () {}

  virtual std::vector<cv::Mat> extractFeatures
    (
    cv::Mat img
    );

  std::vector<cv::UMat> extractFeatures
    (
    cv::UMat img
    );

private:

  float m_smooth;
  int m_smooth_kernel_size;

  cv::UMat smoothImage
    (
    cv::UMat imgLUV
    );
};

#endif


