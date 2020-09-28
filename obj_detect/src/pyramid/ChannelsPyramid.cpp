
#include <pyramid/ChannelsPyramid.h>
#include <channels/Utils.h>
#include <channels/ChannelsExtractorLDCF.h>
#include <opencv2/opencv.hpp>
#include <channels/Utils.h>
#include <cmath>
#include <iostream>

#undef DEBUG
//#define DEBUG

bool
ChannelsPyramid::load(std::string opts)
{
//  bool loadValue = true;
  cv::FileStorage pPyramid;
  bool existOpts = pPyramid.open(opts, cv::FileStorage::READ);
  if(existOpts){
	  int nPerOct = pPyramid["nPerOct"]["data"][0];
      int nOctUp = 0; //pPyramid["nOctUp"]["data"][0];
	  int nApprox = pPyramid["nApprox"]["data"][0];
      //int pad[2] = {pPyramid["pad"]["data"][0], pPyramid["pad"]["data"][1]};
	  int shrink = pPyramid["pChns.shrink"]["data"];

	  m_nOctUp = nOctUp;  
	  m_nPerOct = nPerOct;
	  m_nApprox = nApprox;
	  m_shrink = shrink;
  }
  return existOpts;
}

std::vector<std::vector<cv::Mat>>
ChannelsPyramid::compute
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

  //-------------------------------------------------------------------------

  //GET SCALES AT WHICH TO COMPUTE FEATURES ---------------------------------
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
  ChannelsExtractorLDCF ldcfExtractor(filters, pad, m_shrink);
  //for (const auto& i : isR) // <-- JM: Para solo escalas reales
  for(int i=0; i< nScales; i++) // <-- JM: De momento lo hacemos para todas las escalas (y no solo para las que hay en isR).
  {
    // double s = scales[i - 1]; // <-- JM Para solo escalas reales.
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

  //COMPUTE IMAGE PYRAMID [APPROXIMATE SCALES]-------------------------------
  /*
  for(int i=0; i< isA.size(); i++)
  {
    int x = isA[i] -1;
    int iR =  isN[x];
    int sz_1 = round(sz[0]*scales[x]/m_shrink);
    int sz_2 = round(sz[1]*scales[x]/m_shrink);
    int sz1[2] = {sz_1, sz_2};
    for(int j=0; j < pChnsCompute.size(); j++){
      cv::Mat dataResample = pChnsCompute[j];
      std::vector<cv::Mat> resampleVect;
      for(int k = 0; k < strucData[iR-1].size(); k++){
        cv::Mat resample = ImgResample(strucData[iR-1][k], sz1[0] , sz1[1]); //RATIO
        resampleVect.push_back(resample);
      }
      strucData[x] = resampleVect;
    }
  }
  */

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


/**
 * Funcion getScales. En funcion de los parámetros de entrada retorna un vector con los distintos valores
 * por los que se tiene que escalar la imagen.
 *
 * @param nPerOct: Número de escalas por octava
 * @param nOctUp: Numero de octavas muestreadas para calcular
 * @param minDs: Tamaño mínimo de la imagen
 * @param shrink: Disminucion de la muestra para los canales
 * @param sz: Tamaño de la imagen
 *
 */
int
ChannelsPyramid::getScales
  (
  int nPerOct,
  int nOctUp,
  const cv::Size& minDs,
  int shrink,
  const cv::Size& sz,
  std::vector<double>& scales,
  std::vector<cv::Size2d>& scaleshw
  )
{
  // set each scale s such that max(abs(round(sz*s/shrink)*shrink-sz*s)) is
  // minimized without changing the smaller dim of sz (tricky algebra)
  scales = {};
  scaleshw = {};

  if (!sz.area())
  {
      return 0;
  }

  cv::Size2d ratio(double(sz.width) / double(minDs.width), double(sz.height) / double(minDs.height));
  int nScales = std::floor(double(nPerOct) * (double(nOctUp) + log2(std::min(ratio.width, ratio.height))) + 1.0);

  std::cout << "double(nPerOct) = " << double(nPerOct) << std::endl;
  std::cout << "double(nOctUp) + log2(std::min(ratio.width, ratio.height) = " << double(nOctUp) + log2(std::min(ratio.width, ratio.height)) << std::endl;
  std::cout << "double(nPerOct) * (double(nOctUp) + log2(std::min(ratio.width, ratio.height))) + 1.0 = " << double(nPerOct) * (double(nOctUp) + log2(std::min(ratio.width, ratio.height))) + 1.0 << std::endl;

  double d0 = sz.height, d1 = sz.width;
  if (sz.height >= sz.width)
  {
      std::swap(d0, d1);
  }

  for (int i = 0; i < nScales; i++)
  {
    double s = std::pow(2.0, -double(i) / double(nPerOct) + double(nOctUp));
    double s0 = (std::round(d0 * s / shrink) * shrink - 0.25 * shrink) / d0;
    double s1 = (std::round(d0 * s / shrink) * shrink + 0.25 * shrink) / d0;
    std::pair<double, double> best(0, std::numeric_limits<double>::max());
    for (double j = 0.0; j < 1.0 - std::numeric_limits<double>::epsilon(); j += 0.01)
    {
      double ss = (j * (s1 - s0) + s0);
      double es0 = d0 * ss;
      es0 = std::abs(es0 - std::round(es0 / shrink) * shrink);
      double es1 = d1 * ss;
      es1 = std::abs(es1 - std::round(es1 / shrink) * shrink);
      double es = std::max(es0, es1);
      if (es < best.second)
      {
        best = { ss, es };
      }
    }
    scales.push_back(best.first);
  }

  auto tmp = scales;
  tmp.push_back(0);
  scales.clear();
  for (uint i = 1; i < tmp.size(); i++)
  {
    if (tmp[i] != tmp[i - 1])
    {
      double s = tmp[i - 1];
      scales.push_back(s);

      double x = std::round(double(sz.width) * s / shrink) * shrink / sz.width;
      double y = std::round(double(sz.height) * s / shrink) * shrink / sz.height;
      scaleshw.emplace_back(x, y);
    }
  }

  return 0;
}






