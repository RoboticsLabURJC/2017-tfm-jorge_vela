/** ------------------------------------------------------------------------
 *
 *  @brief Channel feature extractors for LUV color space (using T-API)
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2021/01/29
 *
 *  ------------------------------------------------------------------------ */
#include <iostream>
#include <channels/ChannelsExtractorLUVOpenCL.h>
#include <channels/Utils.h>
#include <opencv2/opencv.hpp>

ChannelsExtractorLUVOpenCL::ChannelsExtractorLUVOpenCL
  (
  bool smooth,
  int smooth_kernel_size
  )
{
  m_smooth = smooth;
  m_smooth_kernel_size = smooth_kernel_size;
}

std::vector<cv::UMat>
ChannelsExtractorLUVOpenCL::smoothImage
  (
  std::vector<cv::UMat> inputImg
  )
{
  std::vector<cv::UMat> channelsLUV_output(3);
  channelsLUV_output[0] = convTri(inputImg[0], m_smooth_kernel_size); //5
  channelsLUV_output[1] = convTri(inputImg[1], m_smooth_kernel_size); //5
  channelsLUV_output[2] = convTri(inputImg[2], m_smooth_kernel_size); //5

  return channelsLUV_output;
}

cv::UMat
ChannelsExtractorLUVOpenCL::smoothImage
  (
  cv::UMat imgLUV
  )
{
  cv::UMat outputImg = convTri(imgLUV, m_smooth_kernel_size); //5

  return outputImg;
}

std::vector<cv::Mat>
ChannelsExtractorLUVOpenCL::extractFeatures
  (
  cv::Mat img
  )
{
  assert(img.type() == CV_8UC3); // TODO: Make it an error check + throw exception

  cv::UMat imgLUV;
  cv::Mat imgLUV_cpu;
  std::vector<cv::Mat> channelsLUV_cpu(3);
  cv::UMat img_float;
  img.convertTo(img_float, CV_32F, 1./255.); // Important to have raw Luv conversion in OpenCV (and not to get it scaled by 255 to fit in 8 bits).

  // (see https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html)
  cv::cvtColor(img_float, imgLUV, cv::COLOR_BGR2Luv);
  cv::divide(imgLUV, 270., imgLUV);
  cv::UMat incMat = cv::UMat(imgLUV.size(), CV_32FC3, cv::Scalar(0.,88./270.,134./270.));
  cv::add(imgLUV, incMat, imgLUV);

  if (m_smooth)
  {
    imgLUV = smoothImage(imgLUV);
  }

  // GPU -> CPU
  imgLUV.copyTo(imgLUV_cpu);
  cv::split(imgLUV_cpu, channelsLUV_cpu);

  return channelsLUV_cpu;

//  cv::split(imgLUV, channelsLUV);

//  // Make the same "normalizations" that P. Dollar's Luv conversion does:
//  cv::multiply(channelsLUV[0], 1.0f/270.0, channelsLUV[0]);
//  cv::add(channelsLUV[1], 88.0, channelsLUV[1]);
//  cv::multiply(channelsLUV[1], 1.0f/270.0, channelsLUV[1]);
//  cv::add(channelsLUV[2], 134.0, channelsLUV[2]);
//  cv::multiply(channelsLUV[2], 1.0f/270.0, channelsLUV[2]);

//  if (m_smooth)
//  {
//    channelsLUV = smoothImage(channelsLUV);
//  }

//  channelsLUV[0].copyTo(channelsLUV_cpu[0]);
//  channelsLUV[1].copyTo(channelsLUV_cpu[1]);
//  channelsLUV[2].copyTo(channelsLUV_cpu[2]);

  return channelsLUV_cpu;
}




