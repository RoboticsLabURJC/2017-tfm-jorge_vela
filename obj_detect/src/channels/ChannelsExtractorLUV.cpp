/** ------------------------------------------------------------------------
 *
 *  @brief Implementation of Channel feature extractors for LUV color space.
 *  @author Jorge Vela
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2019/07/08
 *
 *  ------------------------------------------------------------------------ */

#include <iostream>
#include <channels/ChannelsExtractorLUV.h>
#include <channels/Utils.h>
#include <opencv2/opencv.hpp>

// ------------------- Adapted from Piotr Dollar Matlab Toolbox --------------------
// Constants for rgb2luv conversion and lookup table for y-> l conversion
cv::Mat ChannelsLUVExtractor::bgr2luvSetup
  (
    float scaling_factor, // if image values uint8 -> 1.0/255.0, if float -> 1.0.
    float* mr,
    float* mg,
    float* mb,
    float& minu,
    float& minv,
    float& un,
    float& vn
  )
{
  // Look-Up Table for L values.
  static cv::Mat L_LUT = cv::Mat::zeros(1064, 1, CV_32FC1);
  static bool L_LUT_initialized = false;

  // Constants for conversion
  const float y0 = ((6.0f/29)*(6.0f/29)*(6.0f/29));
  const float a = ((29.0f/3)*(29.0f/3)*(29.0f/3));

  un = 0.197833f;
  vn = 0.468331f;

  mr[0]=0.430574f*scaling_factor;
  mr[1]=0.222015f*scaling_factor;
  mr[2]=0.020183f*scaling_factor;
  mg[0]=0.341550f*scaling_factor;
  mg[1]=0.706655f*scaling_factor;
  mg[2]=0.129553f*scaling_factor;
  mb[0]=0.178325f*scaling_factor;
  mb[1]=0.071330f*scaling_factor;
  mb[2]=0.939180f*scaling_factor;

  float maxi = 1.0f/270;
  minu = -88*maxi;
  minv = -134*maxi;

  // build (padded) lookup table for y->l conversion assuming y in [0,1]
  if (L_LUT_initialized)
  {
    return L_LUT;
  }
  float y, l;
  for(int i=0; i<1025; i++)
  {
    y = (i/1024.0f);
    l = y>y0 ? 116.0f*static_cast<float>(pow(static_cast<double>(y),1.0/3.0))-16 : y*a;
    L_LUT.at<float>(i, 0) = l*maxi;
  }
  for(int i=1025; i<1064; i++)
  {
    L_LUT.at<float>(i, 0) = L_LUT.at<float>(i-1, 0);
  }
  L_LUT_initialized = true;
  return L_LUT;
}

// Convert from rgb to luv
std::vector<cv::Mat> ChannelsLUVExtractor::bgr2luv
  (
  cv::Mat bgr_img,
  float scaling_factor
  )
{
  std::vector<cv::Mat> luv(3);
  std::vector<cv::Mat> bgr(3);
  cv::Size img_sz = bgr_img.size();

  float minu, minv, un, vn, mr[3], mg[3], mb[3];
  cv::Mat L_LUT = bgr2luvSetup(scaling_factor, mr, mg, mb, minu, minv, un, vn);
  luv[0] = cv::Mat::zeros(img_sz, CV_32FC1);
  luv[1] = cv::Mat::zeros(img_sz, CV_32FC1);
  luv[2] = cv::Mat::zeros(img_sz, CV_32FC1);
  cv::split(bgr_img, bgr);

  float x, y, z, l;
  float r, g, b;
  for(int i=0; i < img_sz.height; i++)
  {
    for(int j=0; j < img_sz.width; j++)
    {
      b = bgr[0].at<uint8_t>(i,j);
      g = bgr[1].at<uint8_t>(i,j);
      r = bgr[2].at<uint8_t>(i,j);

      x = mr[0]*r + mg[0]*g + mb[0]*b;
      y = mr[1]*r + mg[1]*g + mb[1]*b;
      z = mr[2]*r + mg[2]*g + mb[2]*b;

      l = L_LUT.at<float>(static_cast<int>(y*1024), 0);
      z = 1/(x + 15*y + 3*z + 1e-35f);
      luv[0].at<float>(i, j) = l;
      luv[1].at<float>(i, j) = l * (13*4*x*z - 13*un) - minu;
      luv[2].at<float>(i, j) = l * (13*9*y*z - 13*vn) - minv;
    }
  }

  return luv;
}

std::vector<cv::Mat> ChannelsLUVExtractor::smoothImage
(
  std::vector<cv::Mat> inputImg
)
{
    std::vector<cv::Mat> channelsLUV_output(3);
    channelsLUV_output[0] = convTri(inputImg[0], m_smooth_kernel_size); //5
    channelsLUV_output[1] = convTri(inputImg[1], m_smooth_kernel_size); //5
    channelsLUV_output[2] = convTri(inputImg[2], m_smooth_kernel_size); //5

    //cv::GaussianBlur(inputImg[0], channelsLUV_output[0], cv::Size(m_smooth_kernel_size, m_smooth_kernel_size), 0,0, cv::BORDER_REFLECT);
    //cv::GaussianBlur(inputImg[1], channelsLUV_output[1], cv::Size(m_smooth_kernel_size, m_smooth_kernel_size), 0,0, cv::BORDER_REFLECT);
    //cv::GaussianBlur(inputImg[2], channelsLUV_output[2], cv::Size(m_smooth_kernel_size, m_smooth_kernel_size), 0,0, cv::BORDER_REFLECT);

    return channelsLUV_output;
}

std::vector<cv::Mat> ChannelsLUVExtractor::extractFeatures
  (
  cv::Mat img
  )
{
  //assert(img.type() == CV_8UC3); // TODO: Make it an error check + throw exception

  std::vector<cv::Mat> channelsLUV(3);
  std::vector<cv::Mat> channelsLUV_normalized(3);
  cv::Mat imgLUV;

  channelsLUV = bgr2luv(img, 1.0f/255.0f);


  if(m_smooth){
    channelsLUV = smoothImage(channelsLUV);
  }

  return channelsLUV;
}

// Not equivalent at all to the P. Dollar one :-(.
//std::vector<cv::Mat> ChannelsLUVExtractor::extractFeaturesOpenCV
//  (
//  cv::Mat img
//  )
//{
//  assert(img.type() == CV_8UC3); // TODO: Make it an error check + throw exception

//  std::vector<cv::Mat> channelsLUV(3);
//  std::vector<cv::Mat> channelsLUV_normalized(3);
//  cv::Mat imgLUV = cv::Mat(img.size(), img.type());

//  // (see https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html)
//  cv::cvtColor(img, imgLUV, cv::COLOR_BGR2Luv);
//  cv::split(imgLUV, channelsLUV);

//  channelsLUV[0].convertTo(channelsLUV_normalized[0], CV_32FC3, 100.0/255.0);
//  channelsLUV[1].convertTo(channelsLUV_normalized[1], CV_32FC3, 354.0/255.0, -134.0);
//  channelsLUV[2].convertTo(channelsLUV_normalized[2], CV_32FC3, 256.0/255.0, -140.0);

//////  // Add some constants in order to do it more similar to P. Dollar's Luv conversion:
////  channelsLUV[1] += 0.325926;
////  channelsLUV[2] += 0.496296;

//  return channelsLUV_normalized;
////  return channelsLUV;
//}

