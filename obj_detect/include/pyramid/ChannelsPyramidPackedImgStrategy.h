/** ------------------------------------------------------------------------
 *
 *  @brief ChannelsPyramidPackedImgStrategy.
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2020/02/17
 *
 *  ------------------------------------------------------------------------ */

#ifndef CHANNELS_PYRAMID_PACKED_IMG_STRATEGY
#define CHANNELS_PYRAMID_PACKED_IMG_STRATEGY

#include <pyramid/ChannelsPyramid.h>
#include <detectors/ClassifierConfig.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>

/** ------------------------------------------------------------------------
 *
 *  @brief Class channels pyramid computation using a packed image (not pyramid).
 *
 *  In this class we compute all channels in the pyramid
 *  with an ImgResample over a packed image + one single call to extractChannels.
 *
 *  ------------------------------------------------------------------------ */
class ChannelsPyramidPackedImgStrategy: public ChannelsPyramid
{
public:
  ChannelsPyramidPackedImgStrategy
    (
    std::string channels_impl_type
    ): ChannelsPyramid(channels_impl_type) {}

  virtual ~ChannelsPyramidPackedImgStrategy
    () {}

  virtual std::vector<std::vector<cv::Mat>> compute
    (
    cv::Mat img,
    std::vector<cv::Mat> filters,
    std::vector<double>& scales,
    std::vector<cv::Size2d>& scaleshw,
    ClassifierConfig clf
    );

protected:

  cv::Size2i
  computePackedImageSize
    (
    const std::vector<cv::Rect2i>& pyr_imgs_rois
    );

  std::vector<cv::Rect2i>
  computePackedPyramidImageROIs
    (
    const std::vector<cv::Size2i>& pyramidImgsSizes
    );

};


#endif
