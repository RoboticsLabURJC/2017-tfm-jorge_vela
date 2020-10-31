/** ------------------------------------------------------------------------
 *
 *  @brief ChannelsPyramidComputeAllStrategy
 *  @author Jorge Vela
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2020/06/01
 *
 *  ------------------------------------------------------------------------ */

#ifndef CHANNELS_PYRAMID_COMPUTE_ALL_STRATEGY
#define CHANNELS_PYRAMID_COMPUTE_ALL_STRATEGY

#include <pyramid/ChannelsPyramid.h>
#include <detectors/ClassifierConfig.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>

/** ------------------------------------------------------------------------
 *
 *  @brief Class channels pyramid computation with approximated channels
 *
 *  In this class we compute all channels in the pyramid
 *  with a combination of ImgResample + extractChannels`.
 *
 *  ------------------------------------------------------------------------ */
class ChannelsPyramidComputeAllStrategy: public ChannelsPyramid
{
public:
  ChannelsPyramidComputeAllStrategy
    (
    std::string channels_impl_type
    ): ChannelsPyramid(channels_impl_type) {}

  virtual ~ChannelsPyramidComputeAllStrategy
  () {}

  virtual std::vector<std::vector<cv::Mat>> compute
    (
    cv::Mat img,
    std::vector<cv::Mat> filters,
    std::vector<double>& scales,
    std::vector<cv::Size2d>& scaleshw,
    ClassifierConfig clf
    );
};


#endif
