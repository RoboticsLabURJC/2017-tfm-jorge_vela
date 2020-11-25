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
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include "sse.hpp"
#include <exception>

#undef DEBUG
//#define DEBUG

void
gradMagOpenCL
  (
  const cv::UMat I, // one channel
  cv::UMat& M,
  cv::UMat& O,
  bool full = true
  )
{
  // Compute always with full: [0,2*pi) (if (!full) -> [0,pi) )

  // Compute [Gx,Gy]=gradient2(I); as in Matlab

  /*cv::Mat kernel_horiz = (cv::Mat_<float>(1,3) << 0.5, 0, -0.5);
  cv::Mat kernel_vert  = (cv::Mat_<float>(3,1) << 0.5, 0, -0.5);
  cv::Mat Gx;
  cv::Mat Gy;
  cv::filter2D(I, Gx, -1, kernel_horiz);
  cv::filter2D(I, Gy, -1, kernel_vert);*/
  //
  cv::UMat Gx, Gy,  Gx2, Gy2;
  cv::Mat kernel_horiz = (cv::Mat_<float>(1,3) << 0.5, 0, -0.5);
  //cv::Mat kernel_vert  = (cv::Mat_<float>(3,1) << 0.5, 0, -0.5);
  cv::filter2D(I, Gx, -1, kernel_horiz);
  cv::filter2D(I, Gy, -1, kernel_horiz.t());

  // Compute gradient mangitude (M)
  /*cv::Mat Gx2, Gy2;
  cv::multiply(Gx, Gx, Gx2);
  cv::multiply(Gy, Gy, Gy2);
  M = Gx2 + Gy2;
  cv::sqrt(M, M);
  M = cv::min(M, 1e10f);*/
  cv::multiply(Gx, Gx, Gx2);
  cv::multiply(Gy, Gy, Gy2);
  cv::add(Gx2, Gy2, M);
  cv::sqrt(M, M);
  //M = M_2.getMat( cv::ACCESS_READ );
  //M = cv::min(M, 1e10f);

  /*// Compute gradient orientation (O)
  cv::multiply(Gx, M, Gx); // Normalize by the gradient magnitude
  cv::multiply(Gy, M, Gy); // Normalize by the gradient magnitude
  cv::phase(Gx, Gy, O, false); // Compute angle in radians in [0, 2pi)*/

  //cv::UMat M_2 = M.getUMat( cv::ACCESS_READ );
  cv::multiply(Gx, M, Gx);
  cv::multiply(Gy, M, Gy); 
  cv::phase(Gx, Gy, O, false);

  //cv::Mat Gy = GyUMat.getMat( cv::ACCESS_READ );
  /*if (!full)
  {
    // Remove M_PI to orientations with Gy < 0 to get orientation in [0, pi)
    cv::Mat isGyNegUint8 = (Gy < 0.0)/255.0; // divide by 255 to get a value of 1 when true
    std::cout << isGyNegUint8.at<int>(0,0) << std::endl;
    cv::Mat isGyNegFloat;
    isGyNegUint8.convertTo(isGyNegFloat, CV_32F);
    cv::Mat toSustract = isGyNegFloat*M_PI;
    O -= toSustract;
  }*/
  if (!full)
  {
    // Remove M_PI to orientations with Gy < 0 to get orientation in [0, pi)
    cv::multiply(Gy, -1, Gy);
    cv::threshold( Gy, Gy, 0, 255, 0 );
    cv::divide( Gy, 255, Gy);

    //cv::UMat isGyNegFloat;
    //GyUMat.convertTo(isGyNegFloat, CV_32F);
    //cv::UMat toSustract;
    cv::multiply(Gy, M_PI, Gy);
    cv::subtract(O, Gy, O);
    //cv::Mat Gy = toSustract.getMat( cv::ACCESS_READ );
    //O -= Gy;
  }
}

std::vector<cv::Mat>
ChannelsExtractorGradMagOpenCL::extractFeatures
  (
  cv::Mat img
  )
{
   cv::UMat IM = img.getUMat( cv::ACCESS_READ );

  int nChannels = img.channels();
  if ((nChannels != 1) && (nChannels != 3))
  {
    throw std::domain_error("Only gray level or BGR (3 channels) images allowed");
  }

  cv::Size orig_sz = img.size();
  std::vector<cv::Mat> channelsGradMag(2);

  cv::UMat channelsGradMag_0 = cv::UMat::zeros(orig_sz.height, orig_sz.width, CV_32FC1);
  cv::UMat channelsGradMag_1 = cv::UMat::zeros(orig_sz.height, orig_sz.width, CV_32FC1);



  //channelsGradMag[0] = cv::Mat::zeros(orig_sz.height, orig_sz.width, CV_32FC1); // M
  //channelsGradMag[1] = cv::Mat::zeros(orig_sz.height, orig_sz.width, CV_32FC1); // O
  if (nChannels == 1)
  {
    cv::UMat img_float;
    IM.convertTo(img_float, CV_32FC1); // important to have continuous memory in img_aux.ptr<float>
    auto start = std::chrono::system_clock::now(); 
    gradMagOpenCL(img_float, channelsGradMag_0, channelsGradMag_1, false);  
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<float,std::milli> duration = end - start;
    std::cout << duration.count() << " time in OpenCL" << std::endl;
    cv::Mat chn0 = channelsGradMag_0.getMat(cv::ACCESS_READ);
    cv::Mat chn1 = channelsGradMag_1.getMat(cv::ACCESS_READ);

    chn0.copyTo(channelsGradMag[0]);
    chn1.copyTo(channelsGradMag[1]);
    //channelsGradMag[0] = channelsGradMag_0.getMat(cv::ACCESS_READ);
    //channelsGradMag[1] = channelsGradMag_1.getMat(cv::ACCESS_READ);
  }
  else if (nChannels == 3)
  {/*
    cv::Mat image_split[3];
    split(img, image_split);

    std::vector<cv::Mat> M_split(3);
    std::vector<cv::Mat> O_split(3);
    for (int i=0; i<3; i++)
    {
      image_split[i].convertTo(image_split[i], CV_32FC1); // important to have continuous memory in img_aux.ptr<float>
      M_split[i] = cv::Mat::zeros(orig_sz.height, orig_sz.width, CV_32FC1);
      O_split[i] = cv::Mat::zeros(orig_sz.height, orig_sz.width, CV_32FC1);
      gradMagOpenCL(image_split[i], M_split[i], O_split[i], false);
    }

    // Compute M matrix
    channelsGradMag[0] = cv::max(cv::max(M_split[0], M_split[1]), M_split[2]);

    // Compute O matrix (on each pixel we put the corresponding value on O_split[i] where
    // the M_split[i] has the maximum value across i=0,1,2).
    cv::Mat M0isLTM1 = (M_split[0] < M_split[1]);
    cv::Mat M0isLTM2 = (M_split[0] < M_split[2]);
    cv::Mat M1isLTM2 = (M_split[1] < M_split[2]);

    cv::Mat M0isMaximum = (~M0isLTM1 & ~M0isLTM2)/255.0;
    M0isMaximum.convertTo(M0isMaximum, CV_32F);
    cv::Mat M1isMaximum = (M0isLTM1 & ~M1isLTM2)/255.0;
    M1isMaximum.convertTo(M1isMaximum, CV_32F);
    cv::Mat M2isMaximum = (M0isLTM2 & M1isLTM2)/255.0;
    M2isMaximum.convertTo(M2isMaximum, CV_32F);

    cv::Mat O0, O1, O2;
    cv::multiply(M0isMaximum, O_split[0], O0);
    cv::multiply(M1isMaximum, O_split[1], O1);
    cv::multiply(M2isMaximum, O_split[2], O2);

    channelsGradMag[1] = O0 + O1 + O2;*/
  }

  if (m_normRad != 0)
  {
    cv::Mat S = convTri(channelsGradMag[0], m_normRad);
    channelsGradMag[0] = channelsGradMag[0] / (S + m_normConst);
  }

#ifdef DEBUG
  cv::imshow("M", channelsGradMag[0]);
  cv::imshow("O", channelsGradMag[1]);
  cv::waitKey();
#endif

  return channelsGradMag;
}
