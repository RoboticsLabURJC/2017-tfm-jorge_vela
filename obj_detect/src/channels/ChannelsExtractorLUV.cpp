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

ChannelsLUVExtractor::ChannelsLUVExtractor
  (
    float smooth,
    int smooth_kernel_size
  )
{
    m_smooth = smooth;
    m_smooth_kernel_size = smooth_kernel_size;

    m_scaling_factor = 1.0f/255.0f;

    // Look-Up Table for L values.
    m_L_LUT = cv::Mat::zeros(1064, 1, CV_32FC1);
    bgr2luvSetup(m_scaling_factor, m_mr, m_mg, m_mb, m_minu, m_minv, m_un, m_vn, m_L_LUT);
}

// Constants for rgb2luv conversion and lookup table for y-> l conversion
void
ChannelsLUVExtractor::bgr2luvSetup
  (
    float scaling_factor, // if image values uint8 -> 1.0/255.0, if float -> 1.0.
    float* mr,
    float* mg,
    float* mb,
    float& minu,
    float& minv,
    float& un,
    float& vn,
    cv::Mat& L_LUT
  )
{
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

  // build lookup table for y->l conversion assuming y in [0,1]
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
}

// Convert from rgb to luv
std::vector<cv::Mat> ChannelsLUVExtractor::bgr2luv
  (
  cv::Mat bgr_img
  )
{
  std::vector<cv::Mat> luv(3);
  std::vector<cv::Mat> bgr(3);
  cv::Size img_sz = bgr_img.size();

//  float minu, minv, un, vn, mr[3], mg[3], mb[3];
//  cv::Mat L_LUT = bgr2luvSetup(scaling_factor, mr, mg, mb, minu, minv, un, vn);
  luv[0] = cv::Mat::zeros(img_sz, CV_32FC1);
  luv[1] = cv::Mat::zeros(img_sz, CV_32FC1);
  luv[2] = cv::Mat::zeros(img_sz, CV_32FC1);
  cv::split(bgr_img, bgr);

#define USE_CVMAT_IMPLEMENTATION
#ifdef USE_CVMAT_IMPLEMANTATION
  cv::Mat b;
  bgr[0].convertTo(b, CV_32FC1);
  cv::Mat g;
  bgr[1].convertTo(g, CV_32FC1);
  cv::Mat r;
  bgr[2].convertTo(r, CV_32FC1);

  cv::Mat x = m_mr[0] * r + m_mg[0] * g + m_mb[0]*b;
  cv::Mat y = m_mr[1] * r + m_mg[1] * g + m_mb[1]*b;
  cv::Mat z = m_mr[2] * r + m_mg[2] * g + m_mb[2]*b;

  cv::Mat l = cv::Mat::zeros(img_sz, CV_32FC1);
  for(int i=0; i < img_sz.height; i++)
  {
    for(int j=0; j < img_sz.width; j++)
    {
      float y_val = y.at<float>(i,j);
      l.at<float>(i,j) = m_L_LUT.at<float>(static_cast<int>(y_val*1024), 0);
    }
  }

  cv::Mat d = (x + 15*y + 3*z + 1e-35f);
  z = 1.0 / d;
  luv[0] = l;
  luv[1] = l.mul(13. * 4. * x.mul(z) - 13.*m_un) - m_minu;
  luv[2] = l.mul(13. * 9. * y.mul(z) - 13.*m_vn) - m_minv;
#else
  // ---------------
  float x_, y_, z_, l_;
  float r_, g_, b_;
  for(int i=0; i < img_sz.height; i++)
  {
    for(int j=0; j < img_sz.width; j++)
    {
      b_ = bgr[0].at<uint8_t>(i,j);
      g_ = bgr[1].at<uint8_t>(i,j);
      r_ = bgr[2].at<uint8_t>(i,j);

//      std::cout << "r_=" << r_ << std::endl;
//      std::cout << "r.at<float>(i,j)=" << r.at<float>(i,j) << std::endl;
//      std::cout << "g_=" << g_ << std::endl;
//      std::cout << "g.at<float>(i,j)=" << g.at<float>(i,j) << std::endl;
//      std::cout << "b_=" << b_ << std::endl;
//      std::cout << "b.at<float>(i,j)=" << b.at<float>(i,j) << std::endl;

      x_ = m_mr[0]*r_ + m_mg[0]*g_ + m_mb[0]*b_;
      y_ = m_mr[1]*r_ + m_mg[1]*g_ + m_mb[1]*b_;
      z_ = m_mr[2]*r_ + m_mg[2]*g_ + m_mb[2]*b_;

//      std::cout << "x_=" << x_ << std::endl;
//      std::cout << "x.at<float>(i,j)=" << x.at<float>(i,j) << std::endl;
//      std::cout << "y_=" << y_ << std::endl;
//      std::cout << "y.at<float>(i,j)=" << y.at<float>(i,j) << std::endl;
//      std::cout << "z_=" << z_ << std::endl;
//      std::cout << "z.at<float>(i,j)=" << z.at<float>(i,j) << std::endl;

      l_ = m_L_LUT.at<float>(static_cast<int>(y_*1024), 0);
      z_ = 1/(x_ + 15*y_ + 3*z_ + 1e-35f);

//      std::cout << "l_=" << l_ << std::endl;
//      std::cout << "l.at<float>(i,j)=" << l.at<float>(i,j) << std::endl;
//      std::cout << "z_=" << z_ << std::endl;
//      std::cout << "z.at<float>(i,j)=" << z.at<float>(i,j) << std::endl;

//      std::cout << "l_=" << l_ << std::endl;
//      std::cout << "luv[0].at<float>(i,j)=" << luv[0].at<float>(i,j) << std::endl;
//      std::cout << "l_ * (13*4*x_*z_ - 13*m_un) - m_minu=" << l_ * (13*4*x_*z_ - 13*m_un) - m_minu << std::endl;
//      std::cout << "luv[1].at<float>(i,j)=" << luv[1].at<float>(i,j) << std::endl;
//      std::cout << "l_ * (13*9*y_*z_ - 13*m_vn) - m_minv=" << l_ * (13*9*y_*z_ - 13*m_vn) - m_minv << std::endl;
//      std::cout << "luv[2].at<float>(i,j)=" << luv[2].at<float>(i,j) << std::endl;

      luv[0].at<float>(i, j) = l_;
      luv[1].at<float>(i, j) = l_ * (13*4*x_*z_ - 13*m_un) - m_minu;
      luv[2].at<float>(i, j) = l_ * (13*9*y_*z_ - 13*m_vn) - m_minv;
    }
  }
#endif

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

  channelsLUV = bgr2luv(img);

  if (m_smooth)
  {
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

