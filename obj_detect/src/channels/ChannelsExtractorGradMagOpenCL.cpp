/** ------------------------------------------------------------------------
 *
 *  @brief Implementation of Channel feature extractors for magnitude and orient gradients.
 *  @author Jorge Vela
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2019/07/08
 *
 *  ------------------------------------------------------------------------ */

#include <iostream>
#include <channels/ChannelsExtractorGradMagOpenCL.h>
#include <channels/Utils.h>
#include "sse.hpp"
#include <exception>

#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>

#undef DEBUG
//#define DEBUG


ChannelsExtractorGradMagOpenCL::ChannelsExtractorGradMagOpenCL
  (
    int normRad,
    float normConst
  ): ChannelsExtractorGradMag(normRad, normConst)
{
  cv::Mat k_horiz = (cv::Mat_<float>(1,3) << 0.5, 0, -0.5);
  cv::Mat k_vert  = (cv::Mat_<float>(3,1) << 0.5, 0, -0.5);
  k_horiz.copyTo(m_kernel_horiz);
  k_vert.copyTo(m_kernel_vert);
};

void
ChannelsExtractorGradMagOpenCL::gradMagOpenCL
  (
  const cv::UMat I, // one channel
  cv::UMat& M,
  cv::UMat& O,
  bool full
  )
{
  // Compute always with full: [0,2*pi) (if (!full) -> [0,pi) )
  // Compute [Gx,Gy]=gradient2(I); as in Matlab
  cv::UMat Gx, Gy, Gx2, Gy2, Gx3, Gy3, GyMult;
  cv::UMat I_float;
  I.convertTo(I_float, CV_32F);
  cv::filter2D(I_float, Gx, -1, m_kernel_horiz);
  cv::filter2D(I_float, Gy, -1, m_kernel_vert);

  cv::multiply(Gx, Gx, Gx2);
  cv::multiply(Gy, Gy, Gy2);
  cv::UMat M_aux;
  cv::add(Gx2, Gy2, M_aux);
  cv::sqrt(M_aux, M); // On place computation is slower?

  cv::multiply(Gx, M, Gx3);
  cv::multiply(Gy, M, Gy3);
  cv::phase(Gx3, Gy3, O, false);

  if (!full)
  {
    // Remove M_PI to orientations with Gy < 0 to get orientation in [0, pi)
    cv::multiply(Gy, -1, Gy);
    cv::threshold(Gy, Gy, 0, 255, 0);
    cv::divide(Gy, 255, Gy); // divide by 255 to get a value of 1 when true

    cv::multiply(Gy, M_PI, GyMult);
    cv::subtract(O, GyMult, O);
  }
}

std::vector<cv::Mat>
ChannelsExtractorGradMagOpenCL::extractFeatures
  (
  cv::Mat img
  )
{
  cv::UMat img_umat;
  // CPU -> GPU
  img.copyTo(img_umat);

  std::vector<cv::UMat> channels_gpu = extractFeatures(img_umat);

  // GPU -> CPU
  std::vector<cv::Mat> channels_cpu;
  for (auto chn: channels_gpu)
  {
    cv::Mat chn_cpu;
    chn.copyTo(chn_cpu);
    channels_cpu.push_back(chn_cpu);
  }

  return channels_cpu;
}

std::vector<cv::UMat>
ChannelsExtractorGradMagOpenCL::extractFeatures
  (
  cv::UMat img
  )
{
  int nChannels = img.channels();
  if ((nChannels != 1) && (nChannels != 3))
  {
    throw std::domain_error("Only gray level or BGR (3 channels) images allowed");
  }

  cv::Size orig_sz = img.size();
  std::vector<cv::UMat> channelsGradMag(2);
  if (nChannels == 1)
  {
    gradMagOpenCL(img, channelsGradMag[0], channelsGradMag[1], false);
  }
  else if (nChannels == 3 )
  {
    cv::UMat M_split[3];
    cv::UMat O_split[3];
    cv::UMat aux;

    std::vector<cv::UMat> img_split(3);
    cv::split(img, img_split);

    // Compute O matrix (on each pixel we put the corresponding value on O_split[i] where
    // the M_split[i] has the maximum value across i=0,1,2).
    gradMagOpenCL(img_split[0], M_split[0], O_split[0], false);
    gradMagOpenCL(img_split[1], M_split[1], O_split[1], false);
    gradMagOpenCL(img_split[2], M_split[2], O_split[2], false);

    // Compute M matrix
    cv::max(M_split[0], M_split[1], M_split[0]);
    cv::max(M_split[0], M_split[2], channelsGradMag[0]);

    cv::UMat M0isLTM1, M0isLTM2 , M1isLTM2;
    cv::compare(M_split[0], M_split[1], M0isLTM1, cv::CMP_LT);
    cv::compare(M_split[0], M_split[2], M0isLTM2, cv::CMP_LT);
    cv::compare(M_split[1], M_split[2], M1isLTM2, cv::CMP_LT);

    cv::UMat notM0isLTM1, notM0isLTM2 , notM1isLTM2;
    cv::bitwise_not(M0isLTM1, notM0isLTM1);
    cv::bitwise_not(M0isLTM2, notM0isLTM2);
    cv::bitwise_not(M1isLTM2, notM1isLTM2);

    // cv::UMat M0isMaximum = (~M0isLTM1 & ~M0isLTM2)/255.0;
    cv::UMat M0isMaximum_255, M0isMaximum;
    cv::bitwise_and(notM0isLTM1, notM0isLTM2, M0isMaximum_255);
    cv::divide(M0isMaximum_255, 255.0, M0isMaximum);
    M0isMaximum.convertTo(M0isMaximum, CV_32F);

    // cv::UMat M1isMaximum = (M0isLTM1 & ~M1isLTM2)/255.0;
    cv::UMat M1isMaximum_255, M1isMaximum;
    cv::bitwise_and(M0isLTM1, notM1isLTM2, M1isMaximum_255);
    cv::divide(M1isMaximum_255, 255.0, M1isMaximum);
    M1isMaximum.convertTo(M1isMaximum, CV_32F);

    // cv::UMat M2isMaximum = (M0isLTM2 & M1isLTM2)/255.0;
    cv::UMat M2isMaximum_255, M2isMaximum;
    cv::bitwise_and(M0isLTM2, M1isLTM2, M2isMaximum_255);
    cv::divide(M2isMaximum_255, 255.0, M2isMaximum);
    M2isMaximum.convertTo(M2isMaximum, CV_32F);

    cv::multiply(M2isMaximum, O_split[2], O_split[2]);
    cv::multiply(M1isMaximum, O_split[1], O_split[1]);
    cv::multiply(M0isMaximum, O_split[0], O_split[0]);

    cv::add(O_split[0], O_split[1], aux);
    cv::add(aux, O_split[2], channelsGradMag[1]);
  }

  if (m_normRad != 0)
  {
//    cv::UMat S = convTri(channelsGradMag[0], m_normRad);
    cv::UMat S;
    convTri(channelsGradMag[0], S, m_normRad);
    cv::add(S, m_normConst, S);
    cv::UMat T;
    cv::divide(channelsGradMag[0], S, T);
    channelsGradMag[0] = T;
  }

  std::vector<cv::UMat> gMag(2);
  channelsGradMag[0].copyTo(gMag[0]);
  channelsGradMag[1].copyTo(gMag[1]);

#ifdef DEBUG
  cv::imshow("M", gMag[0]);
  cv::imshow("O", gMag[1]);
  cv::waitKey();
#endif

  return gMag;
}

