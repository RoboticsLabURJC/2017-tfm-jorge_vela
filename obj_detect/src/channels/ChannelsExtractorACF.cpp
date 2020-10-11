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
#include <channels/ChannelsExtractorGradMagOpenCV.h>
#include <channels/ChannelsExtractorGradMagPDollar.h>
#include <channels/ChannelsExtractorGradHistOpenCV.h>
#include <channels/ChannelsExtractorGradHistPDollar.h>
#include <channels/ChannelsExtractorGradHist.h>
#include <channels/Utils.h>
#include <opencv2/opencv.hpp>


cv::Mat ChannelsExtractorACF::processChannels
(
  cv::Mat image,
  cv::BorderTypes borderType,
  int x,
  int y
  )
{
  image = convTri(image, 1);
  copyMakeBorder(image, image, y, y, x, x, borderType, 0 );
  return image;
}

std::vector<cv::Mat> ChannelsExtractorACF::extractFeatures
  (
  cv::Mat img
  )
{
  int smooth = 1;
  ChannelsExtractorLUV luvExtractor(true, smooth);

  ChannelsExtractorGradMag* pGradMagExtractor;
  if (m_impl_type == "opencv")
  {
    pGradMagExtractor = dynamic_cast<ChannelsExtractorGradMag*>(new ChannelsExtractorGradMagOpenCV(m_gradientMag_normRad, m_gradientMag_normConst));
  }
  else
  {
    pGradMagExtractor = dynamic_cast<ChannelsExtractorGradMag*>(new ChannelsExtractorGradMagPDollar(m_gradientMag_normRad, m_gradientMag_normConst));
  }

  ChannelsExtractorGradHist* pGradHistExtractor;
//  if (m_impl_type == "opencv")
//  {
//    pGradHistExtractor = dynamic_cast<ChannelsExtractorGradHist*>(new ChannelsExtractorGradHistOpenCV(m_gradientHist_binSize,m_gradientHist_nOrients,m_gradientHist_softBin,m_gradientHist_full));
//  }
//  else
//  {
    pGradHistExtractor = dynamic_cast<ChannelsExtractorGradHist*>(new ChannelsExtractorGradHistPDollar(m_gradientHist_binSize,m_gradientHist_nOrients,m_gradientHist_softBin,m_gradientHist_full));
//  }

  //int dChan = img.channels();
  int h = img.size().height;
  int w = img.size().width;

  int crop_h = h % m_shrink;
  int crop_w = w % m_shrink;

  h = h - crop_h;
  w = w - crop_w;

  cv::Rect cropImage = cv::Rect(0, 0, w, h);
  cv::Mat imageCropped = img(cropImage);

  cv::Mat luv_image;
  std::vector<cv::Mat> luvImage;
//  if (m_color_space != "LUV")
//  {
    luvImage = luvExtractor.extractFeatures(imageCropped);
    merge(luvImage, luv_image);
//  }
//  else
//  {
//    luv_image = imageCropped;
//    split(luv_image, luvImage);
//  }
  luv_image = convTri(luv_image, smooth);
  int x = round(m_padding.width / m_shrink);
  int y = round(m_padding.height / m_shrink);
  std::vector<cv::Mat> gMagOrient = pGradMagExtractor->extractFeatures(luv_image);
  std::vector<cv::Mat> gMagHist = pGradHistExtractor->extractFeatures(luv_image, gMagOrient);

  int wResample = w/m_shrink;
  int hResample = h/m_shrink;
  std::vector<cv::Mat> chnsCompute;
  for (cv::Mat luv_i: luvImage)
  {
    cv::Mat resampleLuv = ImgResample(luv_i, wResample, hResample);
    if (m_postprocess_channels)
      resampleLuv = processChannels(resampleLuv, cv::BORDER_REFLECT,x,y);

    chnsCompute.push_back(resampleLuv);

  }

  cv::Mat resampleMag = ImgResample(gMagOrient[0], wResample, hResample);
  if (m_postprocess_channels)
      resampleMag = processChannels(resampleMag, cv::BORDER_CONSTANT,x,y);

  chnsCompute.push_back(resampleMag);

  for(cv::Mat mh_c: gMagHist)
  {
    cv::Mat resampleHist = ImgResample(mh_c, wResample, hResample);
    if (m_postprocess_channels)
      resampleHist = processChannels(resampleHist, cv::BORDER_CONSTANT,x,y);

    chnsCompute.push_back(resampleHist);
  }

  // Remove allocated extractors in dynamic memory:
  delete pGradMagExtractor;
  delete pGradHistExtractor;

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
    int x = round(m_padding.width / m_shrink);
    int y = round(m_padding.height / m_shrink);

    for (uint i=0; i < acf_channels_no_postprocessed.size(); i++)
    {
      cv::Mat c_padded;
      c_padded = convTri(acf_channels_no_postprocessed[i], 1);
      if (i < 3) // LIV channels
      {
        copyMakeBorder( c_padded, c_padded, y, y, x, x, cv::BORDER_REFLECT, 0 );
      }
      else
      {
        copyMakeBorder( c_padded, c_padded, y, y, x, x, cv::BORDER_CONSTANT, 0 );
      }

      postprocessedChannels.push_back(c_padded);
    }
}
