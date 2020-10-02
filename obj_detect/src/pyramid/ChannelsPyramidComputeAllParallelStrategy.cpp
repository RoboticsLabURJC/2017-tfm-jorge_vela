
#include <pyramid/ChannelsPyramidComputeAllParallelStrategy.h>
#include <channels/Utils.h>
#include <channels/ChannelsExtractorLDCF.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/types.hpp>
#include <channels/Utils.h>
#include <cmath>
#include <iostream>

#undef DEBUG
//#define DEBUG

ChannelsPyramidComputeAllParallelStrategy::ChannelsPyramidComputeAllParallelStrategy
  () {};

ChannelsPyramidComputeAllParallelStrategy::~ChannelsPyramidComputeAllParallelStrategy
  () {};

std::vector<std::vector<cv::Mat>>
ChannelsPyramidComputeAllParallelStrategy::compute
  (
  cv::Mat img,
  std::vector<cv::Mat> filters,
  std::vector<double>& scales,
  std::vector<cv::Size2d>& scaleshw
  )
{
  cv::Size sz = img.size();
  cv::Mat imageUse = img;

  getScales(m_nPerOct, m_nOctUp, m_minDs, m_shrink, sz, scales, scaleshw);

#ifdef DEBUG
  std::cout << "--> scales = ";
  for (uint i=0; i < scales.size(); i++)
  {
    std::cout << scales[i] << ", ";
  }
  std::cout << std::endl;
#endif

  int nScales = static_cast<int>(scales.size());
  std::vector<std::vector<cv::Mat>> chnsPyramidData(nScales);
  ChannelsExtractorLDCF ldcfExtractor(filters, m_padding, m_shrink);

  // It is more efficient to compute the
  cv::parallel_for_(cv::Range( 0, nScales ), [&](const cv::Range& r)
  {
    for (int i = r.start; i < r.end; i++)
    {
      double s = scales[i];
      cv::Size sz1;
      sz1.width = round((sz.width * s) / m_shrink) * m_shrink;
      sz1.height = round((sz.height * s) / m_shrink) * m_shrink;

      cv::Mat I1 = ImgResample(imageUse, sz1.width , sz1.height);
      chnsPyramidData[i] = ldcfExtractor.extractFeatures(I1);
    }
  });

  return chnsPyramidData;
}




