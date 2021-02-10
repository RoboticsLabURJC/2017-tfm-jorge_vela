
#include <pyramid/ChannelsPyramidOpenCL.h>
#include <detectors/ClassifierConfig.h>
#include <channels/Utils.h>
#include <channels/ChannelsExtractorLDCF.h>
#include <opencv2/opencv.hpp>
#include <channels/Utils.h>
#include <cmath>
#include <iostream>
#include <algorithm>

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

//std::vector<std::vector<cv::Mat>>
//ChannelsPyramidOpenCL::compute
//  (
//  cv::UMat img,
//  std::vector<cv::Mat> filters,
//  std::vector<double>& scales,
//  std::vector<cv::Size2d>& scaleshw,
//  ClassifierConfig clf
//  )
//{
//  cv::Size sz = img.size();
//  getScales(clf.nPerOct, clf.nOctUp, clf.minDs, clf.shrink, sz, scales, scaleshw);

//#ifdef DEBUG
//  std::cout << "--> scales = ";
//  for (uint i=0; i < scales.size(); i++)
//  {
//    std::cout << scales[i] << ", ";
//  }
//  std::cout << std::endl;
//#endif

//  int nScales = static_cast<int>(scales.size());
//  std::vector<std::vector<cv::UMat>> chnsPyramidData(nScales);
//  std::vector<cv::UMat> pChnsCompute;
//  ChannelsExtractorLDCF ldcfExtractor(filters, clf, "opencl");// clf.padding, clf.shrink, clf.gradMag.normRad, clf.gradMag.normConst, clf.gradHist.binSize, clf.gradHist.nOrients, clf.gradHist.softBin,clf.gradHist.full);
//  for(int i=0; i< nScales; i++)
//  {
//    double s = scales[i];
//    cv::Size sz1;
//    sz1.width = round((sz.width * s) / clf.shrink) * clf.shrink;
//    sz1.height = round((sz.height * s) / clf.shrink) * clf.shrink;

//    cv::UMat I1 = ImgResample(img, sz1.width , sz1.height);
//    chnsPyramidData[i] = ldcfExtractor.extractFeatures(I1);
//  }

//  // GPU -> CPU
//  std::vector<std::vector<cv::Mat>> chnsPyramidData_cpu(nScales);
//  for (int i=0; i < nScales; i++)
//  {
//    std::vector<cv::Mat> chnsPyramidData_cpu_i(chnsPyramidData[i].size());
//    for (int j=0; j < chnsPyramidData[i].size(); j++)
//    {
//      cv::Mat chn;
//      chnsPyramidData[i][j].copyTo(chnsPyramidData_cpu_i[j]);
//    }
//    chnsPyramidData_cpu[i] = chnsPyramidData_cpu_i;
//  }

//  return chnsPyramidData_cpu;
//}

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
  ChannelsExtractorLDCF ldcfExtractor(filters, clf, "opencl");

  // Setup the ROIs within the packed image of every image in the pyramid
  std::vector<cv::Size> pyramid_imgs_sizes;
  std::vector<cv::Size> pyramid_chns_sizes;
  for (int i=0; i< nScales; i++)
  {
    double s = scales[i];
    pyramid_imgs_sizes.emplace_back(round((sz.width * s) / clf.shrink) * clf.shrink,
                                    round((sz.height * s) / clf.shrink) * clf.shrink);

    pyramid_chns_sizes.emplace_back(round(round(0.5 * sz.width * s) / clf.shrink),
                                    round(round(0.5 * sz.height * s) / clf.shrink));

  }

  // Get the ROIs of a big packed image to put the resized images into
  std::vector<cv::Rect2i> pyr_imgs_rois = computePackedPyramidImageROIs(pyramid_imgs_sizes);
  std::vector<cv::Rect2i> pyr_chns_rois = computePackedPyramidImageROIs(pyramid_chns_sizes);

  // Create the packed image with filled with zeros.
  cv::Size packed_img_size = computePackedImageSize(pyr_imgs_rois);
  cv::UMat packed_img = cv::UMat::zeros(packed_img_size, img.type());

#ifdef DEBUG
  int kk=0;
  std::cout << "======================" << std::endl;
  for (auto r: pyr_imgs_rois)
  {
    std::cout << "pyr_imgs_rois[" << kk << "] = " << r << std::endl;
    kk++;
  }
#endif

  // Resize the input image to the scales
  for (auto roi: pyr_imgs_rois)
  {
    cv::UMat roi_img = cv::UMat(packed_img, roi);
    cv::resize(img, roi_img, cv::Size(roi.width, roi.height), 0, 0, cv::INTER_AREA);
  }

  // We only need to compute the channels over the packed image
  std::vector<cv::UMat> chnsCompute;
  chnsCompute = ldcfExtractor.extractFeatures(packed_img);

  // GPU -> CPU
#ifdef DEBUG
  cv::imshow("packed image", packed_img);
  cv::waitKey();
#endif

  std::vector<cv::Mat> chnsCompute_cpu(chnsCompute.size());
  for (int j=0; j < chnsCompute.size(); j++)
  {
    chnsCompute[j].copyTo(chnsCompute_cpu[j]);
  }

  std::vector<std::vector<cv::Mat>> chnsPyramidData_cpu(nScales);
  for (int i=0; i < nScales; i++)
  {
    std::vector<cv::Mat> chnsPyramidData_cpu_i(chnsCompute.size());
    for (int j=0; j < chnsCompute.size(); j++)
    {
#ifdef DEBUG
      cv::imshow("results", chnsCompute[j]);
      cv::waitKey();
#endif
//      cv::UMat chnROI = cv::UMat(chnsCompute[j], pyr_chns_rois[i]);
      cv::Mat chnROI = cv::Mat(chnsCompute_cpu[j], pyr_chns_rois[i]);
#ifdef DEBUG
      cv::imshow("chnROI", chnROI);
      cv::waitKey();
#endif
      chnROI.copyTo(chnsPyramidData_cpu_i[j]);
    }
    chnsPyramidData_cpu[i] = chnsPyramidData_cpu_i;
  }

  return chnsPyramidData_cpu;
}

// Adapted from: https://github.com/hunter-packages/ogles_gpgpu/blob/hunter/ogles_gpgpu/common/proc/pyramid.cpp
cv::Size
ChannelsPyramidOpenCL::computePackedImageSize
  (
  const std::vector<cv::Rect2i>& pyr_imgs_rois
  )
{
  int width = 0;
  int height = 0;

//  pyr_imgs_rois = computePackedPyramidImageROIs(pyramidImgsSizes);
  for (const auto& c : pyr_imgs_rois)
  {
    width = std::max(width, c.x + c.width);
    height = std::max(height, c.y + c.height);
  }

  return cv::Size(width, height);
}

// Adapted from: https://github.com/hunter-packages/ogles_gpgpu/blob/hunter/ogles_gpgpu/common/proc/pyramid.cpp
std::vector<cv::Rect2i>
ChannelsPyramidOpenCL::computePackedPyramidImageROIs
  (
  const std::vector<cv::Size>& pyramid_imgs_size
  )
{
  std::vector<cv::Rect2i> packed;

  int x = 0, y = 0;
  bool half = false;

  // Decrease going forward:
  int i = 0;
  for (; (i < pyramid_imgs_size.size()) && !half; i++)
  {
    packed.emplace_back(x, y, pyramid_imgs_size[i].width, pyramid_imgs_size[i].height);
    x += pyramid_imgs_size[i].width;
    if (pyramid_imgs_size[i].height * 2 < pyramid_imgs_size[0].height)
    {
      half = true;
    }
  }

  // Decrease going backward -- now x becomes the right edge
  int t, l, r, b = pyramid_imgs_size[0].height;
  for (; i < pyramid_imgs_size.size(); i++)
  {
    r = x;
    l = r - pyramid_imgs_size[i].width;
    t = b - pyramid_imgs_size[i].height;
    packed.emplace_back(l, t, pyramid_imgs_size[i].width, pyramid_imgs_size[i].height);
    x -= pyramid_imgs_size[i].width;
  }

  return packed;
}
