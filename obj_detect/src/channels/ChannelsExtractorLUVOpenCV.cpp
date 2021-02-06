/** ------------------------------------------------------------------------
 *
 *  @brief Implementation of Channel feature extractors for LUV color space.
 *  @author Jorge Vela
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2020/10/31
 *
 *  ------------------------------------------------------------------------ */

#include <iostream>
#include <channels/ChannelsExtractorLUVOpenCV.h>
#include <channels/Utils.h>
#include <opencv2/opencv.hpp>

ChannelsExtractorLUVOpenCV::ChannelsExtractorLUVOpenCV
  (
  bool smooth,
  int smooth_kernel_size
  )
{
  m_smooth = smooth;
  m_smooth_kernel_size = smooth_kernel_size;
}

cv::Mat
ChannelsExtractorLUVOpenCV::smoothImage
  (
  cv::Mat inputImg
  )
{
  cv::Mat outputImg = convTri(inputImg, m_smooth_kernel_size); //5

  return outputImg;
}

std::vector<cv::Mat>
ChannelsExtractorLUVOpenCV::extractFeatures
  (
  cv::Mat img
  )
{
  assert(img.type() == CV_8UC3); // TODO: Make it an error check + throw exception

  cv::Mat imgLUV;
  std::vector<cv::Mat> channelsLUV(3);
  cv::Mat img_float;
  img.convertTo(img_float, CV_32F, 1./255.); // Important to have raw Luv conversion in OpenCV (and not to get it scaled by 255 to fit in 8 bits).

  // (see https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html)
  cv::cvtColor(img_float, imgLUV, cv::COLOR_BGR2Luv);
  imgLUV /= 270.;
  cv::Mat incMat = cv::Mat(imgLUV.size(), CV_32FC3, cv::Scalar(0.,88./270.,134./270.));
  imgLUV += incMat;

  if (m_smooth)
  {
    imgLUV = smoothImage(imgLUV);
  }
  cv::split(imgLUV, channelsLUV);

  return channelsLUV;
}


