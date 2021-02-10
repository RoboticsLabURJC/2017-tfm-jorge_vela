
#include <pyramid/ChannelsPyramidApproximatedParallelStrategy.h>
#include <channels/Utils.h>
#include <channels/ChannelsExtractorLDCF.h>
#include <channels/ChannelsExtractorACF.h>
#include <opencv2/opencv.hpp>
#include <channels/Utils.h>
#include <cmath>
#include <iostream>
#include <omp.h>
#undef DEBUG
#include <chrono>
#include <random>

std::vector<std::vector<cv::Mat>>
ChannelsPyramidApproximatedParallelStrategy::compute
  (
  cv::Mat img,
  std::vector<cv::Mat> filters,
  std::vector<double>& scales,
  std::vector<cv::Size2d>& scaleshw,
  ClassifierConfig clf
  )
{
  cv::Size sz = img.size();

  // GET SCALES AT WHICH TO COMPUTE FEATURES ---------------------------------
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
  std::vector<int> isR, isA, isN(nScales, 0), *isRA[2] = { &isR, &isA };
  for (int i = 0; i < nScales; i++)
  {
    isRA[(i % (clf.nApprox + 1)) > 0]->push_back(i + 1);
  }

  
  std::vector<int> isH((isR.size() + 1), 0);
  isH.back() = nScales;
  for (int i = 0; i < std::max(int(isR.size()) - 1, 0); i++)
  {
    isH[i + 1] = (isR[i] + isR[i + 1]) / 2;
  }

  for (uint i = 0; i < isR.size(); i++)
  {
    for (int j = isH[i]; j < isH[i + 1]; j++)
    {
      isN[j] = isR[i];
    }
  }

  std::vector<std::vector<cv::Mat>> chnsPyramidDataACF(nScales);
  bool postprocess_acf_channels = false; // here we do not postprocess ACF channels!!
  ChannelsExtractorACF acfExtractor(clf, postprocess_acf_channels, m_channels_impl_type);

  cv::parallel_for_(cv::Range( 0, isR.size() ), [&](const cv::Range& r)
  {
    for (int i = r.start; i < r.end; i++)
    {
      double s = scales[isR[i] - 1];      
      cv::Size sz1;
      sz1.width = round((sz.width * s) / clf.shrink) * clf.shrink;
      sz1.height = round((sz.height * s) / clf.shrink) * clf.shrink;

      cv::Mat I1;
      if (sz == sz1)
      {
        I1 = img;
      }
      else
      {
        ImgResample(img, I1, sz1.width , sz1.height);
      }
      chnsPyramidDataACF[isR[i]-1] = acfExtractor.extractFeatures(I1);
    }
  });

  // Idea From: https://github.com/elucideye/acf/blob/master/src/lib/acf/acf/chnsPyramid.cpp
  // The per scale/type operations are easily parallelized, but with a parallel_for approach
  // using simple uniform slicing will tend to starve some threads due to the nature of the
  // pyramid layout.  Randomizing the scale indices should do better.  More optimal strategies
  // may exist with further testing (work stealing, etc).
  auto isAIndex = isA;
  std::shuffle(isAIndex.begin(), isAIndex.end(), std::mt19937(std::random_device()()));

  cv::parallel_for_(cv::Range( 0, int(isAIndex.size()) ), [&](const cv::Range& r)
  {
    for (int i = r.start; i < r.end; i++)
    {
      int i1 = isA[i] - 1;
      int iR = isN[i1] - 1;

      cv::Size2f sz1(round(sz.width*scales[i1]/clf.shrink),
                     round(sz.height*scales[i1]/clf.shrink));
      std::vector<cv::Mat> resampleVect(acfExtractor.getNumChannels());
      for (int k = 0; k < acfExtractor.getNumChannels(); k++)
      {
        int type_of_channel_index=2;
        if (k < 4)
        {
          type_of_channel_index = (k/3) ? 1:0;
        }
        
        float ratio = pow((scales[i1]/scales[iR]),-clf.lambdas[type_of_channel_index]);
        ImgResample(chnsPyramidDataACF[iR][k], resampleVect[k], sz1.width , sz1.height, "antialiasing", ratio);
      }
      chnsPyramidDataACF[i1] = resampleVect;
    }
  });

  // Now we can filter the channels to get the LDCF ones.
  ChannelsExtractorLDCF ldcfExtractor(filters, clf);//, clf.padding, clf.shrink, clf.gradMag.normRad, clf.gradMag.normConst, clf.gradHist.binSize, clf.gradHist.nOrients, clf.gradHist.softBin,clf.gradHist.full); //clf.padding, clf.shrink, m_gradientMag_normRad, m_gradientMag_normConst, m_gradientHist_binSize, m_gradientHist_nOrients,m_gradientHist_softBin,m_gradientHist_full);
  std::vector<std::vector<cv::Mat>> chnsPyramidData(nScales);

  // Idea From: https://github.com/elucideye/acf/blob/master/src/lib/acf/acf/chnsPyramid.cpp
  // The per scale/type operations are easily parallelized, but with a parallel_for approach
  // using simple uniform slicing will tend to starve some threads due to the nature of the
  // pyramid layout.  Randomizing the scale indices should do better.  More optimal strategies
  // may exist with further testing (work stealing, etc).
  const auto scalesIndex = create_random_indices(nScales);
  cv::parallel_for_(cv::Range( 0, chnsPyramidDataACF.size()), [&](const cv::Range& r)
  {
    for (int i = r.start; i < r.end; i++)
    {
      // Postprocess the non-postprocessed ACF channels
      std::vector<cv::Mat> acfChannels;
      acfExtractor.postProcessChannels(chnsPyramidDataACF[scalesIndex[i]], acfChannels);

      // Compute LDCF from the postprocessed ACF channels
      chnsPyramidData[scalesIndex[i]] = ldcfExtractor.extractFeaturesFromACF(acfChannels);
    }
  });

  return chnsPyramidData;
}




