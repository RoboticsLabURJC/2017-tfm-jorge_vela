
/** ------------------------------------------------------------------------
 *
 *  @brief Channel feature extractors for LUV color space.
 *  @author Jorge Vela
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2019/07/08
 *
 *  ------------------------------------------------------------------------ */

#ifndef CHANNELS_EXTRACTOR_LUV
#define CHANNELS_EXTRACTOR_LUV

#include <opencv2/opencv.hpp>
#include <vector>

class ChannelsExtractorLUV
{
public:
  ChannelsExtractorLUV
    (
    bool smooth = true,
    int smooth_kernel_size = 1
    )
  {
    m_smooth = smooth;
    m_smooth_kernel_size = m_smooth_kernel_size;
  };

  ~ChannelsExtractorLUV
    () {}

  virtual std::vector<cv::Mat> extractFeatures
    (
      cv::Mat img
    ) = 0;

  virtual std::vector<cv::Mat> bgr2luv
    (
      cv::Mat bgr_img
    ) = 0;

  static std::shared_ptr<ChannelsExtractorLUV>
  createExtractor
    (
      std::string extractor_type = "pdollar",
      bool smooth = true,
      int smooth_kernel_size = 1
    );

protected:
  bool m_smooth;
  int m_smooth_kernel_size;
};

#endif
