/** ------------------------------------------------------------------------
 *
 *  @brief ChannelsPyramidOpenCL.
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2020/02/07
 *
 *  ------------------------------------------------------------------------ */

#ifndef CHANNELS_PYRAMID_OPENCL
#define CHANNELS_PYRAMID_OPENCL

#include <detectors/ClassifierConfig.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>

/** ------------------------------------------------------------------------
 *
 *  @brief Class channels pyramid computation with T-API (OpenCL)
 *
 *  In this class we compute all channels in the pyramid
 *  with a combination of ImgResample + extractChannels`.
 *
 *  ------------------------------------------------------------------------ */
class ChannelsPyramidOpenCL
{
public:
  ChannelsPyramidOpenCL
    () {}

  ~ChannelsPyramidOpenCL
    () {}

  int getScales
    (
    int nPerOct,
    int nOctUp,
    const cv::Size& minDs,
    int shrink,
    const cv::Size& sz,
    std::vector<double>& scales,
    std::vector<cv::Size2d>& scaleshw
    );

  std::vector<std::vector<cv::Mat>>
  compute
    (
    cv::UMat img,
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
