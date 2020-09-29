
#include <pyramid/ChannelsPyramidComputeAllStrategy.h>
#include <channels/Utils.h>
#include <channels/ChannelsExtractorLDCF.h>
#include <opencv2/opencv.hpp>
#include <channels/Utils.h>
#include <cmath>
#include <iostream>

#undef DEBUG
//#define DEBUG

ChannelsPyramidComputeAllStrategy::ChannelsPyramidComputeAllStrategy
  () {};

ChannelsPyramidComputeAllStrategy::~ChannelsPyramidComputeAllStrategy
  () {};

std::vector<std::vector<cv::Mat>>
ChannelsPyramidComputeAllStrategy::compute
  (
  cv::Mat img,
  std::vector<cv::Mat> filters
  )
{
  int smooth = 1;
  cv::Size sz = img.size();
  cv::Size minDs;
  minDs.width = 84; // <--- TODO: JM: Esto debería de venir del fichero del detector.
  minDs.height = 48; // <--- TODO: JM: Esto debería de venir de fichero del detector
  cv::Size pad;
  pad.width = 6; //12; //12; // <--- TODO: JM: Esto debería de venir del fichero del detector.
  pad.height = 4; //6; //6; // <--- TODO: JM: Esto debería de venir de fichero del detector

  //int lambdas = {};
  cv::Mat imageUse = img;

  std::vector<double> scales;
  std::vector<cv::Size2d> scaleshw;
  getScales(m_nPerOct, m_nOctUp, minDs, m_shrink, sz, scales, scaleshw);

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
  ChannelsExtractorLDCF ldcfExtractor(filters, pad, m_shrink);
  for(int i=0; i< nScales; i++)
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

    if ((s == 0.5) && (m_nApprox > 0 || m_nPerOct == 1))
    {
      imageUse = I1;
    }

    chnsPyramidData[i] = ldcfExtractor.extractFeatures(I1);
  }

  return chnsPyramidData;
}




