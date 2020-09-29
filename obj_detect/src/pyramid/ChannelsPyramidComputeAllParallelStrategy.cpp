
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

ChannelsPyramidComputeAllParrallelStrategy::ChannelsPyramidComputeAllParrallelStrategy
  () {};

ChannelsPyramidComputeAllParrallelStrategy::~ChannelsPyramidComputeAllParrallelStrategy
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
ChannelsPyramidComputeAllParrallelStrategy::compute
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


  cv::parallel_for_({ 0, nScales }, [&](const cv::Range& r) {
    for (int i = r.start; i < r.end; i++)
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

      chnsPyramidData[i] = ldcfExtractor.extractFeatures(I1);
      }
    });

  return chnsPyramidData;
}




