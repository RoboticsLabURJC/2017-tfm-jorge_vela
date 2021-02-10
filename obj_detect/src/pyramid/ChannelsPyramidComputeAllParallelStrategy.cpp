
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

std::vector<std::vector<cv::Mat>>
ChannelsPyramidComputeAllParallelStrategy::compute
  (
  cv::Mat img,
  std::vector<cv::Mat> filters,
  std::vector<double>& scales,
  std::vector<cv::Size2d>& scaleshw,
  ClassifierConfig clf
  )
{
  cv::Size sz = img.size();
  cv::Mat imageUse = img;

  getScales(clf.nPerOct, clf.nOctUp, clf.minDs, clf.shrink, sz, scales, scaleshw);

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
  ChannelsExtractorLDCF ldcfExtractor(filters, clf, m_channels_impl_type);

  // Idea From: https://github.com/elucideye/acf/blob/master/src/lib/acf/acf/chnsPyramid.cpp
  // The per scale/type operations are easily parallelized, but with a parallel_for approach
  // using simple uniform slicing will tend to starve some threads due to the nature of the
  // pyramid layout.  Randomizing the scale indices should do better.  More optimal strategies
  // may exist with further testing (work stealing, etc).
  const auto scalesIndex = create_random_indices(nScales);

  cv::parallel_for_(cv::Range( 0, int(scalesIndex.size()) ), [&](const cv::Range& r)
//  cv::parallel_for_(cv::Range( 0, nScales ), [&](const cv::Range& r)
  {
    for (int i = r.start; i < r.end; i++)
    {
//      double s = scales[i];
      double s = scales[scalesIndex[i]];
      cv::Size sz1;
      sz1.width = round((sz.width * s) / clf.shrink) * clf.shrink;
      sz1.height = round((sz.height * s) / clf.shrink) * clf.shrink;

      cv::Mat I1;
      ImgResample(imageUse, I1, sz1.width , sz1.height);
      chnsPyramidData[scalesIndex[i]] = ldcfExtractor.extractFeatures(I1);
//      chnsPyramidData[i] = ldcfExtractor.extractFeatures(I1);
    }
  });

  return chnsPyramidData;
}




