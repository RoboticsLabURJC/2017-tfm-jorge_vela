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
  //int smooth = 1;
  ChannelsLUVExtractor luvExtractor( m_clf.luv.smooth, m_clf.luv.smooth_kernel_size);//(true, smooth);

  GradMagExtractor gradMagExtract( m_clf.gradMag.normRad, m_clf.gradMag.normConst); // 5
  GradHistExtractor gradHistExtract(m_clf.gradHist.binSize,m_clf.gradHist.nOrients,m_clf.gradHist.softBin,m_clf.gradHist.full); 

  //int dChan = img.channels();
  int h = img.size().height;
  int w = img.size().width;

  int crop_h = h % m_clf.shrink;
  int crop_w = w % m_clf.shrink;

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
  luv_image = convTri(luv_image,  m_clf.luv.smooth_kernel_size);
  int x = round(m_clf.padding.width / m_clf.shrink);
  int y = round(m_clf.padding.height / m_clf.shrink);
  std::vector<cv::Mat> gMagOrient = gradMagExtract.extractFeatures(luv_image);
  std::vector<cv::Mat> gMagHist = gradHistExtract.extractFeatures(luv_image, gMagOrient);

  int wResample = w/m_clf.shrink;
  int hResample = h/m_clf.shrink;
  std::vector<cv::Mat> chnsCompute;
  //std::vector<ChannelsExtractorACF::channel> chnsCompute;
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

  /*if (!m_postprocess_channels)
  {
    return chnsCompute;
  }

  // Postprocessing of the ACF channels
  std::vector<cv::Mat> postprocessedChannels;
  //for (channel c: chnsCompute)
  for (uint i=0; i < chnsCompute.size(); i++)
  {
    cv::Mat c_padded;
    
    //c_padded = convTri(c.image, 1);
    //if (c.type == "LUV")
    if (i < 12) // LIV channels
    {
      c_padded = chnsCompute[i];
      //copyMakeBorder( c_padded, c_padded, y, y, x, x, cv::BORDER_REFLECT, 0 );
    }
    else
    {
      c_padded = convTri(chnsCompute[i], 1);
      copyMakeBorder( c_padded, c_padded, y, y, x, x, cv::BORDER_CONSTANT, 0 );
    }
     
    postprocessedChannels.push_back(c_padded);
  }*/

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
