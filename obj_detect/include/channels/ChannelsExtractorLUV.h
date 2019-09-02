
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

#include <opencv/cv.hpp>
#include <vector>

class ChannelsLUVExtractor
{
public:
  ChannelsLUVExtractor
    (
    float smooth = false;
    int smooth_kernel_size = 1;
    ) {};

  std::vector<cv::Mat> extractFeatures
    (
    cv::Mat img
    );

// Not equivalent at all to the P. Dollar one :-(.
//  std::vector<cv::Mat> extractFeaturesOpenCV
//    (
//    cv::Mat img
//    );

protected:
  cv::Mat bgr2luvSetup
    (
      float scaling_factor, // if image values uint8 -> 1.0/255.0, if float -> 1.0.
      float* mr,
      float* mg,
      float* mb,
      float& minu,
      float& minv,
      float& un,
      float& vn
    );

  std::vector<cv::Mat> bgr2luv
    (
    cv::Mat bgr_img,
    float scaling_factor // if image values uint8 -> 1.0/255.0, if float -> 1.0.
    );
};

#endif
