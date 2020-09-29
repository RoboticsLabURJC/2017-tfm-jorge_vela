
#include <pyramid/ChannelsPyramidApproximatedStrategy.h>
#include <channels/Utils.h>
#include <channels/ChannelsExtractorLDCF.h>
#include <opencv2/opencv.hpp>
#include <channels/Utils.h>
#include <cmath>
#include <iostream>

#undef DEBUG
//#define DEBUG

ChannelsPyramidApproximatedStrategy::ChannelsPyramidApproximatedStrategy
  () {};

ChannelsPyramidApproximatedStrategy::~ChannelsPyramidApproximatedStrategy
  () {};

std::vector<std::vector<cv::Mat>>
ChannelsPyramidApproximatedStrategy::compute
  (
  cv::Mat img,
  std::vector<cv::Mat> filters,
  std::vector<double>& scales,
  std::vector<cv::Size2d>& scaleshw
  )
{
  int smooth = 1;
  cv::Size sz = img.size();
//  cv::Size minDs;
//  minDs.width = 84; // <--- TODO: JM: Esto debería de venir del fichero del detector.
//  minDs.height = 48; // <--- TODO: JM: Esto debería de venir de fichero del detector
//  cv::Size pad;
//  pad.width = 6; //12; //12; // <--- TODO: JM: Esto debería de venir del fichero del detector.
//  pad.height = 4; //6; //6; // <--- TODO: JM: Esto debería de venir de fichero del detector

  //int lambdas = {};
  cv::Mat imageUse = img;

  //-------------------------------------------------------------------------

  //GET SCALES AT WHICH TO COMPUTE FEATURES ---------------------------------
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

  std::vector<std::vector<cv::Mat>> chnsPyramidData(nScales);
  std::vector<cv::Mat> pChnsCompute;
  ChannelsExtractorLDCF ldcfExtractor(filters, m_padding, m_shrink);
  for (const auto& i : isR) // <-- JM: Para solo escalas reales
  //for(int i=0; i< nScales; i++) // <-- JM: De momento lo hacemos para todas las escalas (y no solo para las que hay en isR).
  {
    // double s = scales[i - 1]; // <-- JM Para solo escalas reales.
    double s = scales[i-1];
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

    chnsPyramidData[i-1] = ldcfExtractor.extractFeatures(I1);
  }

  //COMPUTE IMAGE PYRAMID [APPROXIMATE SCALES]------------------------------- 
  float lambdas[3] = {0.000000, 0.073930, 0.072470};
  for(int i=0; i< isA.size(); i++)
  {
    int x = isA[i] -1;
    int iR =  isN[x];
    double s = scales[x];
    int sz_1 = round(sz.width*scales[i]/m_shrink); //round((sz.width * s) / m_shrink) * m_shrink; //
    int sz_2 = round(sz.height*scales[i]/m_shrink); //round((sz.height * s) / m_shrink) * m_shrink; //
    int pad_x = round(m_padding.width / m_shrink);
    int pad_y = round(m_padding.height / m_shrink);

    sz_1 = round(round((sz_1+pad_x))*0.5);
    sz_2 = round(round((sz_2+pad_y))*0.5);

    int sz1[2] = {sz_1, sz_2};
      std::vector<cv::Mat> resampleVect;
      for(int k = 0; k < chnsPyramidData[iR-1].size(); k++){
        int lambda;
        if(k%10 < 3){
          lambda = 0;
        }else if(k%10 == 3){
          lambda = 1;
        }else if(k&10 > 3){
          lambda = 2;
        }
        float ratio=pow((scales[i]/scales[iR]),-lambdas[lambda]);
        cv::Mat resample = ImgResample(chnsPyramidData[iR-1][k], sz1[0] , sz1[1], "antialiasing", ratio); //RATIO
        resampleVect.push_back(resample);
    }
    chnsPyramidData[x] = resampleVect;
  }

  /*
  std::vector<cv::Mat> channelsConcat;
  int x = pad.width / m_shrink;
  int y = pad.height / m_shrink;
  for(int i = 0; i < nScales; i++)
  {
      cv::Mat concat;
      merge(chnsPyramidData[i], concat);
      concat = convTri(concat, 1);
      copyMakeBorder( concat, concat, y, y, x, x, cv::BORDER_REFLECT, 0 );
      //copyMakeBorder( concat, concat, 2, 2, 3, 3, cv::BORDER_REPLICATE, 0 );
      channelsConcat.push_back(concat);
  }
  */

  return chnsPyramidData;
}




