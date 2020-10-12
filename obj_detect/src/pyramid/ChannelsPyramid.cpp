
#include <pyramid/ChannelsPyramid.h>
#include <channels/Utils.h>
#include <channels/ChannelsExtractorLDCF.h>
#include <opencv2/opencv.hpp>
#include <channels/Utils.h>
#include <cmath>
#include <iostream>

#undef DEBUG
//#define DEBUG

ChannelsPyramid::ChannelsPyramid
  () {};

ChannelsPyramid::~ChannelsPyramid
  () {};

/*
bool
ChannelsPyramid::load(std::string opts)
{
  cv::FileStorage pPyramid;
  bool existOpts = pPyramid.open(opts, cv::FileStorage::READ);

  if (existOpts)
  {
    m_padding.width = pPyramid["pad"]["data"][1]; //6; //
    m_padding.height = pPyramid["pad"]["data"][0]; //4; //
    m_nOctUp = pPyramid["nOctUp"]["data"][0]; //0; //
    m_nPerOct = pPyramid["nPerOct"]["data"][0]; //3; //
    m_nApprox = pPyramid["nApprox"]["data"][0]; //2; //
    m_shrink = pPyramid["pChns.shrink"]["data"];

    m_gradientMag_normRad = pPyramid["pChns.pGradMag"]["normRad"]; //5;
    m_gradientMag_normConst = pPyramid["pChns.pGradMag"]["normConst"]; //0.005;


    m_gradientHist_binSize =  pPyramid["pChns.pGradHist"]["enabled"]; //2;
    m_gradientHist_nOrients =  pPyramid["pChns.pGradHist"]["nOrients"]; //6;
    m_gradientHist_softBin =  pPyramid["pChns.pGradHist"]["softBin"]; //1;
    m_gradientHist_full =  false ; //pPyramid["pChns.pGradHist"]["full"]; //0;


    int lambdasSize = pPyramid["lambdas"]["cols"];
    for(int i = 0; i < lambdasSize; i++)
      m_lambdas.push_back((float)pPyramid["lambdas"]["data"][i]);

    // TODO: Cargar del fichero!!
    m_minDs.width = pPyramid["minDs"]["data"][1]; //84; // <--- TODO: JM: Esto debería de venir del fichero del detector.
    m_minDs.height = pPyramid["minDs"]["data"][0]; // 48; // <--- TODO: JM: Esto debería de venir de fichero del detector
  }
  return existOpts;
}*/

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






