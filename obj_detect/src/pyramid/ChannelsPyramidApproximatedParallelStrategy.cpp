
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
//#define DEBUG
#include <chrono>

ChannelsPyramidApproximatedParallelStrategy::ChannelsPyramidApproximatedParallelStrategy
  () {};

ChannelsPyramidApproximatedParallelStrategy::~ChannelsPyramidApproximatedParallelStrategy
  () {};

std::vector<std::vector<cv::Mat>>
ChannelsPyramidApproximatedParallelStrategy::compute
  (
  cv::Mat img,
  std::vector<cv::Mat> filters,
  std::vector<double>& scales,
  std::vector<cv::Size2d>& scaleshw
  )
{
  cv::Size sz = img.size();
  //cv::Mat imageUse = img;

  // GET SCALES AT WHICH TO COMPUTE FEATURES ---------------------------------
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
  std::vector<int> isR, isA, isN(nScales, 0), *isRA[2] = { &isR, &isA };
  for (int i = 0; i < nScales; i++)
  {
    isRA[(i % (m_nApprox + 1)) > 0]->push_back(i + 1);
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
  //std::vector<cv::Mat> pChnsCompute;
  bool postprocess_acf_channels = false; // here we do not postprocess ACF channels!!
  ChannelsExtractorACF acfExtractor(m_padding, m_shrink, postprocess_acf_channels, m_gradientMag_normRad, m_gradientMag_normConst, m_gradientHist_binSize, m_gradientHist_nOrients,m_gradientHist_softBin,m_gradientHist_full);
  //uint i;
  cv::parallel_for_(cv::Range( 0, isR.size() ), [&](const cv::Range& r)
  {
    for (int i = r.start; i < r.end; i++)
    {
      double s = scales[isR[i] - 1];      
      cv::Size sz1;
      sz1.width = round((sz.width * s) / m_shrink) * m_shrink;
      sz1.height = round((sz.height * s) / m_shrink) * m_shrink;

      cv::Mat I1;
      if (sz == sz1)
      {
        I1 = img;
      }
      else
      {
        I1 = ImgResample(img, sz1.width , sz1.height);
      }

      if ((s == 0.5) && (m_nApprox > 0 || m_nPerOct == 1))
      {
        img = I1;
      }

      chnsPyramidDataACF[isR[i]-1] = acfExtractor.extractFeatures(I1);
      }
  });

  cv::parallel_for_(cv::Range( 0, isA.size() ), [&](const cv::Range& r)
  {
    for (int i = r.start; i < r.end; i++)
    {
      int i1 = isA[i] - 1;
      int iR = isN[i1] - 1;

      cv::Size2f sz1(round(sz.width*scales[i1]/m_shrink),
                     round(sz.height*scales[i1]/m_shrink));
      std::vector<cv::Mat> resampleVect(acfExtractor.getNumChannels());
      for (int k = 0; k < acfExtractor.getNumChannels(); k++)
      {
        int type_of_channel_index=2;
        if(k < 4)
          type_of_channel_index = (k/3) ? 1:0;
        
        float ratio = pow((scales[i1]/scales[iR]),-m_lambdas[type_of_channel_index]);
        cv::Mat resample = ImgResample(chnsPyramidDataACF[iR][k], sz1.width , sz1.height, "antialiasing", ratio); 
        resampleVect[k] = resample;//.push_back(resample);
      }
      chnsPyramidDataACF[i1] = resampleVect;
    }
  });
  //  COMPUTE IMAGE PYRAMID [APPROXIMATE SCALES]-------------------------------  //printf("helloooo1\n");
  /*for (const auto& i : isA)// for(int i=0; i< isA.size(); i++) // 
  {
    int i1 = i - 1;
    int iR = isN[i1] - 1;

    cv::Size2f sz1(round(sz.width*scales[i1]/m_shrink),
                   round(sz.height*scales[i1]/m_shrink));
    std::vector<cv::Mat> resampleVect(acfExtractor.getNumChannels());

    cv::parallel_for_(cv::Range( 0, acfExtractor.getNumChannels() ), [&](const cv::Range& r)
      {
        for (int k = r.start; k < r.end; k++)
        {
          int type_of_channel_index=2;
          if(k < 4)
            type_of_channel_index = (k/3) ? 1:0;
          
          float ratio = pow((scales[i1]/scales[iR]),-m_lambdas[type_of_channel_index]);
          cv::Mat resample = ImgResample(chnsPyramidDataACF[iR][k], sz1.width , sz1.height, "antialiasing", ratio); 
          resampleVect[k] = resample;//.push_back(resample);
        }
      });
    chnsPyramidDataACF[i1] = resampleVect;
  }*/

  // Now we can filter the channels to get the LDCF ones.
  ChannelsExtractorLDCF ldcfExtractor(filters, m_padding, m_shrink, m_gradientMag_normRad, m_gradientMag_normConst, m_gradientHist_binSize, m_gradientHist_nOrients,m_gradientHist_softBin,m_gradientHist_full);
  std::vector<std::vector<cv::Mat>> chnsPyramidData(nScales);

  cv::parallel_for_(cv::Range( 0, chnsPyramidDataACF.size()), [&](const cv::Range& r)
  {
    for (uint i = r.start; i < r.end; i++)
    {
      // Postprocess the non-postprocessed ACF channels
      std::vector<cv::Mat> acfChannels;
      acfExtractor.postProcessChannels(chnsPyramidDataACF[i], acfChannels);

      // Compute LDCF from the postprocessed ACF channels
      chnsPyramidData[i] = ldcfExtractor.extractFeaturesFromACF(acfChannels);
    }
  });

  return chnsPyramidData;
}




