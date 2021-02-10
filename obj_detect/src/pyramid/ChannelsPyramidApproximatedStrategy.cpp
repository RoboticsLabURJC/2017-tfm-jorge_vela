
#include <pyramid/ChannelsPyramidApproximatedStrategy.h>
#include <channels/Utils.h>
#include <channels/ChannelsExtractorLDCF.h>
#include <channels/ChannelsExtractorACF.h>
#include <opencv2/opencv.hpp>
#include <channels/Utils.h>
#include <cmath>
#include <iostream>
#include <chrono>

#undef DEBUG
//#define DEBUG


std::vector<std::vector<cv::Mat>>
ChannelsPyramidApproximatedStrategy::compute
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
  getScales(clf.nPerOct, clf.nOctUp, clf.minDs, clf.shrink, sz, scales, scaleshw); //getScales(m_nPerOct, m_nOctUp, m_minDs, m_shrink, sz, scales, scaleshw);

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
    isRA[(i % (clf.nApprox + 1)) > 0]->push_back(i + 1); //m_nApprox
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

  // Full computation for the real scales (ImResample+extractFeatures)
  for (const auto& i : isR)
  {
    double s = scales[i-1];
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

    if ((s == 0.5) && (clf.nApprox > 0 || clf.nPerOct == 1))
    {
      img = I1; 
    }

    chnsPyramidDataACF[i-1] = acfExtractor.extractFeatures(I1);
  }

  //  Compute approximated scales in the pyramid
  for (const auto& i : isA)
  {
    int i1 = i - 1;
    int iR = isN[i1] - 1;

    cv::Size2f sz1(round(sz.width*scales[i1]/clf.shrink),
                   round(sz.height*scales[i1]/clf.shrink));
    std::vector<cv::Mat> resampleVect(acfExtractor.getNumChannels());
    for (int k = 0; k < acfExtractor.getNumChannels(); k++)
    {
      int type_of_channel_index=2;
      if (k < 4)
        type_of_channel_index = (k/3) ? 1:0;
        /*int type_of_channel_index;
        if (k > 3)
          type_of_channel_index = 2; // LUV channels
        else if (k == 3)
          type_of_channel_index = 1; // Magnitude of gradient channel
        else //if (k > 3)
          type_of_channel_index = 0; // HoG channels.*/
      
      float ratio = pow((scales[i1]/scales[iR]),-clf.lambdas[type_of_channel_index]);
      ImgResample(chnsPyramidDataACF[iR][k], resampleVect[k], sz1.width , sz1.height, "antialiasing", ratio);
    }
    chnsPyramidDataACF[i1] = resampleVect;
  }


  // Now we can filter the channels to get the LDCF ones.
  ChannelsExtractorLDCF ldcfExtractor(filters, clf);
  std::vector<std::vector<cv::Mat>> chnsPyramidData(nScales);

  for (uint i=0; i < chnsPyramidDataACF.size(); i++)
  {
    // Postprocess the non-postprocessed ACF channels
    std::vector<cv::Mat> acfChannels;
    acfExtractor.postProcessChannels(chnsPyramidDataACF[i], acfChannels);

    // Compute LDCF from the postprocessed ACF channels
    chnsPyramidData[i] = ldcfExtractor.extractFeaturesFromACF(acfChannels);
  } 
  return chnsPyramidData;
}




