/** ------------------------------------------------------------------------
 *
 *  @brief Implementation of Aggregated Channels Features (ACF)
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2020/09/25
 *
 *  ------------------------------------------------------------------------ */

#include <iostream>
#include <channels/ChannelsExtractorACF.h>
#include <channels/ChannelsExtractorLUV.h>
#include <channels/ChannelsExtractorGradMag.h>
#include <channels/ChannelsExtractorGradHist.h>
#include <channels/ChannelsExtractorLUVOpenCL.h>
#include <channels/ChannelsExtractorGradMagOpenCL.h>
#include <channels/ChannelsExtractorGradHistOpenCL.h>
#include <channels/ChannelsExtractorGradHist.h>
#include <channels/Utils.h>
#include <opencv2/opencv.hpp>
#include <memory>

ChannelsExtractorACF::ChannelsExtractorACF
  (
  ClassifierConfig clf,
  bool postprocess_channels,
  std::string impl_type
  )
{
  m_impl_type = impl_type;
  m_clf = clf;
  m_postprocess_channels = postprocess_channels;

  m_pGradMagExtractor = ChannelsExtractorGradMag::createExtractor(m_impl_type,
                                                                  m_clf.gradMag.normRad,
                                                                  m_clf.gradMag.normConst);

  m_pGradHistExtractor = ChannelsExtractorGradHist::createExtractor(m_impl_type,
                                                                    m_clf.gradHist.binSize,
                                                                    m_clf.gradHist.nOrients,
                                                                    m_clf.gradHist.softBin,
                                                                    m_clf.gradHist.full);

  m_pLUVExtractor = ChannelsExtractorLUV::createExtractor(m_impl_type,
                                                          m_clf.luv.smooth,
                                                          m_clf.luv.smooth_kernel_size);
};

void
ChannelsExtractorACF::processChannel
(
  cv::Mat& image,
  cv::BorderTypes borderType,
  int x,
  int y
  )
{
//  image = convTri(image, 1);
  convTri(image, image, 1);
  copyMakeBorder(image, image, y, y, x, x, borderType, 0 );
}

std::vector<cv::Mat>
ChannelsExtractorACF::extractFeatures
  (
  cv::Mat img
  )
{
  int h = img.size().height;
  int w = img.size().width;

  int crop_h = h % m_clf.shrink;
  int crop_w = w % m_clf.shrink;

  h = h - crop_h;
  w = w - crop_w;

  cv::Rect cropImage = cv::Rect(0, 0, w, h);
  cv::Mat imageCropped = img(cropImage);

  cv::Mat luvImage;
  std::vector<cv::Mat> luvChannels(3);

  luvChannels = m_pLUVExtractor->extractFeatures(imageCropped);
  merge(luvChannels, luvImage);

//  luvImage = convTri(luvImage,  m_clf.luv.smooth_kernel_size);
  convTri(luvImage, luvImage, m_clf.luv.smooth_kernel_size);
  int x = round(m_clf.padding.width / m_clf.shrink);
  int y = round(m_clf.padding.height / m_clf.shrink);
  std::vector<cv::Mat> gMagOrient = m_pGradMagExtractor->extractFeatures(luvImage);
  std::vector<cv::Mat> gMagHist = m_pGradHistExtractor->extractFeatures(luvImage, gMagOrient);

  int wResample = w/m_clf.shrink;
  int hResample = h/m_clf.shrink;
  std::vector<cv::Mat> chnsCompute;
  for (cv::Mat luv_i: luvChannels)
  {
    cv::Mat resampleLuv;
    ImgResample(luv_i, resampleLuv, wResample, hResample);
    if (m_postprocess_channels)
    {
      processChannel(resampleLuv, cv::BORDER_REFLECT,x,y);
    }

    chnsCompute.push_back(resampleLuv);
  }

  cv::Mat resampleMag;
  ImgResample(gMagOrient[0], resampleMag, wResample, hResample);
  if (m_postprocess_channels)
  {
    processChannel(resampleMag, cv::BORDER_CONSTANT,x,y);
  }

  chnsCompute.push_back(resampleMag);

  for(cv::Mat mh_c: gMagHist)
  {
    cv::Mat resampleHist;
    ImgResample(mh_c, resampleHist, wResample, hResample);
    if (m_postprocess_channels)
    {
      processChannel(resampleHist, cv::BORDER_CONSTANT,x,y);
    }

    chnsCompute.push_back(resampleHist);
  }

  return chnsCompute;
}

void
ChannelsExtractorACF::postProcessChannels
  (
  const std::vector<cv::Mat>& acf_channels_no_postprocessed,
  std::vector<cv::Mat>& postprocessedChannels
  )
{
  // Postprocessing of the ACF channels
  int x = round(m_clf.padding.width / m_clf.shrink);
  int y = round(m_clf.padding.height / m_clf.shrink);
  postprocessedChannels.clear();

  for (uint i=0; i < acf_channels_no_postprocessed.size(); i++)
  {
    cv::Mat c_padded;
//    c_padded = convTri(acf_channels_no_postprocessed[i], 1);
    convTri(acf_channels_no_postprocessed[i], c_padded, 1);
//    if (i < 3) // LUV channels
//    {
      copyMakeBorder( c_padded, c_padded, y, y, x, x, cv::BORDER_REFLECT, 0 );
//    }
//    else
//    {
//      copyMakeBorder( c_padded, c_padded, y, y, x, x, cv::BORDER_CONSTANT, 0 );
//    }

    postprocessedChannels.push_back(c_padded);
  }
}

// -----------------------------------------------------------
// ------ UMat as input and outputs
// -----------------------------------------------------------
void
ChannelsExtractorACF::processChannel
(
  cv::UMat& image,
  cv::BorderTypes borderType,
  int x,
  int y
  )
{
//  image = convTri(image, 1);
  convTri(image, image, 1);
  copyMakeBorder(image, image, y, y, x, x, borderType, 0 );
}

std::vector<cv::UMat>
ChannelsExtractorACF::extractFeatures
  (
  cv::UMat img
  )
{
  ChannelsExtractorGradMagOpenCL gradMagExtractor(m_clf.gradMag.normRad,
                                                  m_clf.gradMag.normConst);

  ChannelsExtractorGradHistOpenCL gradHistExtractor(m_clf.gradHist.binSize,
                                                    m_clf.gradHist.nOrients,
                                                    m_clf.gradHist.softBin,
                                                    m_clf.gradHist.full);

  ChannelsExtractorLUVOpenCL LUVExtractor(m_clf.luv.smooth,
                                          m_clf.luv.smooth_kernel_size);


  int h = img.size().height;
  int w = img.size().width;

  int crop_h = h % m_clf.shrink;
  int crop_w = w % m_clf.shrink;

  h = h - crop_h;
  w = w - crop_w;

  cv::Rect cropImage = cv::Rect(0, 0, w, h);
  cv::UMat imageCropped = img(cropImage);

  cv::UMat luvImage;
  std::vector<cv::UMat> luvChannels(3);
  luvChannels = LUVExtractor.extractFeatures(imageCropped);
  merge(luvChannels, luvImage);
//  luvImage = convTri(luvImage,  m_clf.luv.smooth_kernel_size);
  convTri(luvImage, luvImage, m_clf.luv.smooth_kernel_size);
  int x = round(m_clf.padding.width / m_clf.shrink);
  int y = round(m_clf.padding.height / m_clf.shrink);
  std::vector<cv::UMat> gMagOrient = gradMagExtractor.extractFeatures(luvImage);
  std::vector<cv::UMat> gMagHist = gradHistExtractor.extractFeatures(luvImage, gMagOrient);

  int wResample = w/m_clf.shrink;
  int hResample = h/m_clf.shrink;
  std::vector<cv::UMat> chnsCompute;
  for (cv::UMat luv_i: luvChannels)
  {
    cv::UMat resampleLuv;
    ImgResample(luv_i, resampleLuv, wResample, hResample);
    if (m_postprocess_channels)
    {
      processChannel(resampleLuv, cv::BORDER_REFLECT,x,y);
    }

    chnsCompute.push_back(resampleLuv);
  }

  cv::UMat resampleMag;
  ImgResample(gMagOrient[0], resampleMag, wResample, hResample);
  if (m_postprocess_channels)
  {
    processChannel(resampleMag, cv::BORDER_CONSTANT,x,y);
  }

  chnsCompute.push_back(resampleMag);

  for(cv::UMat mh_c: gMagHist)
  {
    cv::UMat resampleHist;
    ImgResample(mh_c, resampleHist, wResample, hResample);
    if (m_postprocess_channels)
    {
      processChannel(resampleHist, cv::BORDER_CONSTANT,x,y);
    }

    chnsCompute.push_back(resampleHist);
  }

  return chnsCompute;
}


void
ChannelsExtractorACF::postProcessChannels
  (
  const std::vector<cv::UMat>& acf_channels_no_postprocessed,
  std::vector<cv::UMat>& postprocessedChannels
  )
{
  // Postprocessing of the ACF channels
  int x = round(m_clf.padding.width / m_clf.shrink);
  int y = round(m_clf.padding.height / m_clf.shrink);
  postprocessedChannels.clear();

  for (uint i=0; i < acf_channels_no_postprocessed.size(); i++)
  {
    cv::UMat c_padded;
//    c_padded = convTri(acf_channels_no_postprocessed[i], 1);
    convTri(acf_channels_no_postprocessed[i], c_padded, 1);
//    if (i < 3) // LUV channels
//    {
      copyMakeBorder( c_padded, c_padded, y, y, x, x, cv::BORDER_REFLECT, 0 );
//    }
//    else
//    {
//      copyMakeBorder( c_padded, c_padded, y, y, x, x, cv::BORDER_CONSTANT, 0 );
//    }

    postprocessedChannels.push_back(c_padded);
  }
}

