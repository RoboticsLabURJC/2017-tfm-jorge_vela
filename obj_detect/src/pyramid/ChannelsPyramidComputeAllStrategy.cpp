
#include <pyramid/ChannelsPyramidComputeAllStrategy.h>
#include <channels/Utils.h>
#include <channels/ChannelsExtractorLDCF.h>
#include <opencv2/opencv.hpp>
#include <channels/Utils.h>
#include <cmath>
#include <iostream>

#undef DEBUG
//#define DEBUG

std::vector<std::vector<cv::Mat>>
ChannelsPyramidComputeAllStrategy::compute
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
  std::vector<cv::Mat> pChnsCompute;
  ChannelsExtractorLDCF ldcfExtractor(filters, clf, m_channels_impl_type);
  for(int i=0; i< nScales; i++)
  {
    double s = scales[i];
    cv::Size sz1;
    sz1.width = round((sz.width * s) / clf.shrink) * clf.shrink;
    sz1.height = round((sz.height * s) / clf.shrink) * clf.shrink;

    cv::Mat I1;
    if (sz == sz1)
    {
      I1 = imageUse;
    }
    else
    {
      ImgResample(imageUse, I1, sz1.width , sz1.height);
    }

    if ((s == 0.5) && (clf.nApprox > 0 || clf.nPerOct == 1))
    {
      imageUse = I1;
    }

    chnsPyramidData[i] = ldcfExtractor.extractFeatures(I1);
  }

  return chnsPyramidData;
}




