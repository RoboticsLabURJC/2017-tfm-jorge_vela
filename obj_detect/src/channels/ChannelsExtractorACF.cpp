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
  //std::vector<ChannelsExtractorACF::channel> chnsCompute;
  for (cv::Mat luv_i: luvImage)
  {
    cv::Mat resampleLuv = ImgResample(luv_i, w/m_shrink, h/m_shrink);
    chnsCompute.push_back(resampleLuv);
//    channel ch1;
//    ch1.image = resampleLuv;
//    ch1.type = "LUV";
//    chnsCompute.push_back(ch1);
  }

  cv::Mat resampleMag = ImgResample(gMagOrient[0], w/m_shrink, h/m_shrink);
  chnsCompute.push_back(resampleMag);
//  channel ch2;
//  ch2.image = resampleMag;
//  ch2.type = "GMAG";
//  chnsCompute.push_back(ch2);


  for(cv::Mat mh_c: gMagHist)
  {
    cv::Mat resampleHist = ImgResample(mh_c, w/m_shrink, h/m_shrink);
    chnsCompute.push_back(resampleHist);
//    channel ch3;
//    ch3.image = resampleHist;
//    ch3.type = "GHIST";
//    chnsCompute.push_back(ch3);
  }

  if (!m_postprocess_channels)
  {
    return chnsCompute;
  }

  // Postprocessing of the ACF channels
  std::vector<cv::Mat> postprocessedChannels;
  int x = round(m_padding.width / m_shrink);
  int y = round(m_padding.height / m_shrink);

//  for (channel c: chnsCompute)
  for (uint i=0; i < chnsCompute.size(); i++)
  {
    cv::Mat c_padded;
    c_padded = convTri(chnsCompute[i], 1);
//    c_padded = convTri(c.image, 1);
//    if (c.type == "LUV")
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

  return postprocessedChannels;
}


std::vector<cv::Mat>
ChannelsExtractorACF::postProcessChannels
  (
  std::vector<cv::Mat>& acf_channels_no_postprocessed
  )
{
    // Postprocessing of the ACF channels
    std::vector<cv::Mat> postprocessedChannels;
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

    return postprocessedChannels;
}
