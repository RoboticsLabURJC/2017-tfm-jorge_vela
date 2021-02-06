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

std::vector<cv::Mat>
ChannelsExtractorLUVOpenCV::smoothImage
  (
  std::vector<cv::Mat> inputImg
  )
{
  std::vector<cv::Mat> channelsLUV_output(3);
  channelsLUV_output[0] = convTri(inputImg[0], m_smooth_kernel_size); //5
  channelsLUV_output[1] = convTri(inputImg[1], m_smooth_kernel_size); //5
  channelsLUV_output[2] = convTri(inputImg[2], m_smooth_kernel_size); //5

  return channelsLUV_output;
}

//std::vector<cv::Mat>
//ChannelsExtractorLUVOpenCV::extractFeatures
//  (
//  cv::Mat img
//  )
//{
//  assert(img.type() == CV_8UC3); // TODO: Make it an error check + throw exception

//  std::vector<cv::Mat> channelsLUV(3);
//  std::vector<cv::Mat> channelsLUV_normalized(3);
//  cv::Mat img_float;
//  img.convertTo(img_float, CV_32F, 1./255.); // Important to have raw Luv conversion in OpenCV (and not to get it scaled by 255 to fit in 8 bits).
//  cv::Mat imgLUV; // = cv::Mat::zeros(img.size(), CV_32F);

//  // (see https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html)
//  cv::cvtColor(img_float, imgLUV, cv::COLOR_BGR2Luv);
//  cv::split(imgLUV, channelsLUV);

//  // Make the same "normalizations" that P. Dollar's Luv conversion does:
//  channelsLUV[0] /= 270.0;
//  channelsLUV[1] += 88.0;
//  channelsLUV[1] /= 270.0;
//  channelsLUV[2] += 134.0;
//  channelsLUV[2] /= 270.0;

//  if (m_smooth)
//  {
//    channelsLUV = smoothImage(channelsLUV);
//  }

//  return channelsLUV;
//}


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


