
/** ------------------------------------------------------------------------
 *
 *  @brief Channel feature extractors for LUV color space.
 *  @author Jorge Vela
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2019/07/08
 *
 *  ------------------------------------------------------------------------ */

#ifndef CHANNELS_EXTRACTOR_LUV_PDOLLAR
#define CHANNELS_EXTRACTOR_LUV_PDOLLAR

#include <channels/ChannelsExtractorLUV.h>
#include <opencv2/opencv.hpp>
#include <vector>

class ChannelsExtractorLUVPDollar: public ChannelsExtractorLUV
{
public:

  ChannelsExtractorLUVPDollar
    (
    bool smooth = true,
    int smooth_kernel_size = 1
    );

  ~ChannelsExtractorLUVPDollar
    () {}

  virtual std::vector<cv::Mat> extractFeatures
    (
      cv::Mat img
    );

  virtual std::vector<cv::Mat> bgr2luv
    (
      cv::Mat bgr_img
    );

private:

  float m_smooth;
  int m_smooth_kernel_size;

  cv::Mat m_L_LUT;
  float m_minu, m_minv, m_un, m_vn, m_mr[3], m_mg[3], m_mb[3];
  float m_scaling_factor;

  void bgr2luvSetup
    (
      float scaling_factor, // if image values uint8 -> 1.0/255.0, if float -> 1.0.
      float* mr,
      float* mg,
      float* mb,
      float& minu,
      float& minv,
      float& un,
      float& vn,
      cv::Mat& L_LUT
    );

  std::vector<cv::Mat> smoothImage
    (
    std::vector<cv::Mat> channelsLUV_input
    );
};

#endif
