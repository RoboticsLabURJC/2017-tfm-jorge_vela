
#include <pyramid/ChannelsPyramidComputeAllParallelStrategy.h>
#include <channels/Utils.h>
#include <channels/ChannelsExtractorLDCF.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/types.hpp>
#include <channels/Utils.h>
#include <cmath>
#include <iostream>
#include <functional>
#include <numeric>
#include <random>

#undef DEBUG
//#define DEBUG

ChannelsPyramidComputeAllParallelStrategy::ChannelsPyramidComputeAllParallelStrategy
  () {};

ChannelsPyramidComputeAllParallelStrategy::~ChannelsPyramidComputeAllParallelStrategy
  () {};

inline std::vector<int>
create_random_indices
  (
  int n
  )
{
    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), std::mt19937(std::random_device()()));
    return indices;
}

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
  std::vector<cv::Mat> pChnsCompute;
  std::vector<cv::Mat> resampledImages;
  ChannelsExtractorLDCF ldcfExtractor(filters, m_padding, m_shrink);

  for (int i = 0; i < nScales; i++)
  {
    double s = scales[i];
    cv::Size sz1;
    sz1.width = round((sz.width * s) / m_shrink) * m_shrink;
    sz1.height = round((sz.height * s) / m_shrink) * m_shrink;

    cv::Mat I1;
    if (sz == sz1)
    {
      I1 = imageUse;
    }
    else
    {
      I1 = ImgResample(imageUse, sz1.width , sz1.height);
    }
    I1 = ImgResample(img, sz1.width , sz1.height);
    resampledImages.push_back(I1);
  }

  cv::parallel_for_({ 0, nScales }, [&](const cv::Range& r)
  {
    for (int i = r.start; i < r.end; i++)
    {
      chnsPyramidData[i] = ldcfExtractor.extractFeatures(resampledImages[i]);
    }
  }
  ); // parallel_for_


  return chnsPyramidData;
}



