
#include <pyramid/ChannelsPyramidOpenCL.h>
#include <detectors/ClassifierConfig.h>
#include <channels/Utils.h>
#include <channels/ChannelsExtractorLDCF.h>
#include <opencv2/opencv.hpp>
#include <channels/Utils.h>
#include <cmath>
#include <iostream>

#undef DEBUG
//#define DEBUG

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
ChannelsPyramidOpenCL::getScales
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

std::vector<std::vector<cv::Mat>>
ChannelsPyramidOpenCL::compute
  (
  cv::UMat img,
  std::vector<cv::Mat> filters,
  std::vector<double>& scales,
  std::vector<cv::Size2d>& scaleshw,
  ClassifierConfig clf
  )
{
  cv::Size sz = img.size();
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
  std::vector<std::vector<cv::UMat>> chnsPyramidData(nScales);
  std::vector<cv::UMat> pChnsCompute;
  ChannelsExtractorLDCF ldcfExtractor(filters, clf, "opencl");// clf.padding, clf.shrink, clf.gradMag.normRad, clf.gradMag.normConst, clf.gradHist.binSize, clf.gradHist.nOrients, clf.gradHist.softBin,clf.gradHist.full);
  for(int i=0; i< nScales; i++)
  {
    double s = scales[i];
    cv::Size sz1;
    sz1.width = round((sz.width * s) / clf.shrink) * clf.shrink;
    sz1.height = round((sz.height * s) / clf.shrink) * clf.shrink;

    cv::UMat I1 = ImgResample(img, sz1.width , sz1.height);
    chnsPyramidData[i] = ldcfExtractor.extractFeatures(I1);
  }

  // GPU -> CPU
  std::vector<std::vector<cv::Mat>> chnsPyramidData_cpu(nScales);
  for (int i=0; i < nScales; i++)
  {
    std::vector<cv::Mat> chnsPyramidData_cpu_i(chnsPyramidData[i].size());
    for (int j=0; j < chnsPyramidData[i].size(); j++)
    {
      cv::Mat chn;
      chnsPyramidData[i][j].copyTo(chnsPyramidData_cpu_i[j]);
    }
    chnsPyramidData_cpu[i] = chnsPyramidData_cpu_i;
  }

  return chnsPyramidData_cpu;
}








