/** ------------------------------------------------------------------------
 *
 *  @brief ChannelsPyramidApproximatedStrategy
 *  @author Jorge Vela
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2020/09/29
 *
 *  ------------------------------------------------------------------------ */

#ifndef CHANNELS_PYRAMID_APPROXIMATED_PARALLEL_STRATEGY
#define CHANNELS_PYRAMID_APPROXIMATED_PARALLEL_STRATEGY

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
 *  In this class we implement the strategy given by P.Dollar in his original
 *  Matlab toolbox. In this case some channels in the pyramid are computed
 *  with a combination of ImgResample + extractChannels while most of them
 *  are approximated by doing ImgResample of the computed channels.
 *
 *  In this case we perform the computation in a cv::parallel_loop_.
 *
 *  ------------------------------------------------------------------------ */
class ChannelsPyramidApproximatedParallelStrategy: public ChannelsPyramid
{
public:
  ChannelsPyramidApproximatedParallelStrategy
    (
    std::string channels_impl_type
    ): ChannelsPyramid(channels_impl_type) {}

  virtual ~ChannelsPyramidApproximatedParallelStrategy
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
