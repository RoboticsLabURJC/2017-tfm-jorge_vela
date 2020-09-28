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
#include <channels/Utils.h>
#include <opencv2/opencv.hpp>

std::vector<cv::Mat> ChannelsExtractorACF::extractFeatures
  (
  cv::Mat img
  )
{
  int smooth = 1;
  ChannelsLUVExtractor luvExtractor{false, smooth};
  GradMagExtractor gradMagExtract{5};
  GradHistExtractor gradHistExtract{2,6,1,0}; //{4,6,1,1}; // <--- JM: Cuidado!! Estos parámetros dependerán del clasificador entrenado?

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

  std::vector<cv::Mat> gMagOrient = gradMagExtract.extractFeatures(luv_image);
  std::vector<cv::Mat> gMagHist = gradHistExtract.extractFeatures(luv_image, gMagOrient);

  std::vector<cv::Mat> chnsCompute;
  for (cv::Mat luv_i: luvImage)
  {
    cv::Mat resampleLuv = ImgResample(luv_i, w/m_shrink, h/m_shrink);
    chnsCompute.push_back(resampleLuv);
  }

  cv::Mat resampleMag = ImgResample(gMagOrient[0], w/m_shrink, h/m_shrink);
  chnsCompute.push_back(resampleMag);

  for(cv::Mat mh_c: gMagHist)
  {
    cv::Mat resampleHist = ImgResample(mh_c, w/m_shrink, h/m_shrink);
    chnsCompute.push_back(resampleHist);
  }

  // Preprocessing of the ACF channels
  std::vector<cv::Mat> preprocessedChannels;
  int x = round(m_padding.width / m_shrink);
  int y = round(m_padding.height / m_shrink);

  for (cv::Mat c: chnsCompute)
  {
    cv::Mat c_padded;
    c_padded = convTri(c, 1);
    copyMakeBorder( c_padded, c_padded, y, y, x, x, cv::BORDER_CONSTANT, 0 );
    preprocessedChannels.push_back(c_padded);
  }

  return preprocessedChannels;
}

