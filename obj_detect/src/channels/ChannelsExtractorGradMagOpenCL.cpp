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

  cv::filter2D(I, Gx, -1, k_horiz);
  cv::filter2D(I, Gy, -1, k_horiz.t());

  cv::multiply(Gx, Gx, Gx2);
  cv::multiply(Gy, Gy, Gy2);
  cv::add(Gx2, Gy2, M);
  cv::sqrt(M, M);

  cv::multiply(Gx, M, Gx3);
  cv::multiply(Gy, M, Gy3); 
  cv::phase(Gx3, Gy3, O, false);

  if (!full)
  {
    cv::multiply(Gy, -1, Gy);
    cv::threshold( Gy, Gy, 0, 255, 0 );
    cv::divide( Gy, 255, Gy);

    cv::UMat toSustract;
    cv::multiply(Gy, M_PI, GyMult);
    cv::subtract(O, GyMult, O);
  }
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
  cv::UMat channelsGradMag_0 = cv::UMat::zeros(orig_sz.height, orig_sz.width, CV_32FC1);
  cv::UMat channelsGradMag_1 = cv::UMat::zeros(orig_sz.height, orig_sz.width, CV_32FC1);  
  std::vector<cv::UMat> gMag;

  cv::Mat kernel_horiz = (cv::Mat_<float>(1,3) << 0.5, 0, -0.5);
  k_horiz = kernel_horiz.getUMat(cv::ACCESS_READ);

  if (nChannels == 1 )
  {
    gradMagOpenCL(img,channelsGradMag_0, channelsGradMag_1,false);
  }
  else if (nChannels == 3 )
  {
    gradMagOpenCL(img,channelsGradMag_0, channelsGradMag_1,false);

    cv::Mat imgSplit[3];
    split(img.getMat(cv::ACCESS_READ), imgSplit); 

    cv::UMat M_split[3];
    cv::UMat O_split[3];
    /*cv::Mat M_split[3];
    split(channelsGradMag_0.getMat(cv::ACCESS_READ), M_split); 

    cv::Mat O_split[3];
    split(channelsGradMag_1.getMat(cv::ACCESS_READ), O_split); 

    cv::UMat M_sp0 = M_split[0].getUMat(cv::ACCESS_READ);
    cv::UMat M_sp1 = M_split[1].getUMat(cv::ACCESS_READ);
    cv::UMat M_sp2 = M_split[2].getUMat(cv::ACCESS_READ);

    cv::UMat O_sp0 = O_split[0].getUMat(cv::ACCESS_READ);
    cv::UMat O_sp1 = O_split[1].getUMat(cv::ACCESS_READ);
    cv::UMat O_sp2 = O_split[2].getUMat(cv::ACCESS_READ);

    cv::UMat cmp1, cmp2;
    cv::max(M_sp0, M_sp1, cmp1);
    cv::max(cmp1, M_sp2, channelsGradMag_0);*/

    //gMag.push_back(channelsGradMag_0);
    //gMag.push_back(channelsGradMag_1);
    //cv::imshow("A",gMag[0]);
    //cv::waitKey(0);
    /*cv::Mat splitted[3];
    cv::Mat image_split[3];
    cv::Mat img2 = img.getMat(cv::ACCESS_READ);
    split(img2, image_split);
    std::vector<cv::UMat> M_split(3);
    std::vector<cv::UMat> O_split(3);*/

    for (int i=0; i<3; i++)
    {
      //image_split[i].convertTo(image_split[i], CV_32FC1); // important to have continuous memory in img_aux.ptr<float>
      //M_split[i] = cv::UMat::zeros(orig_sz.height, orig_sz.width, CV_32FC1); //cv::Mat::zeros(orig_sz.height, orig_sz.width, CV_32FC1);
      //O_split[i] = cv::UMat::zeros(orig_sz.height, orig_sz.width, CV_32FC1); //cv::Mat::zeros(orig_sz.height, orig_sz.width, CV_32FC1);
      M_split[i] = cv::UMat::zeros(orig_sz.height, orig_sz.width, CV_32FC1);
      O_split[i] = cv::UMat::zeros(orig_sz.height, orig_sz.width, CV_32FC1);  
      gradMagOpenCL(imgSplit[i].getUMat(cv::ACCESS_READ),  M_split[i], O_split[i], false);
    }
   
    //cv::UMat cmp1;
    cv::max(M_split[0], M_split[1], M_split[0]);
    cv::max(M_split[0], M_split[2], channelsGradMag_0); 


    cv::UMat M0isLTM1, M0isLTM2 , M1isLTM2;
    cv::compare(M_split[0], M_split[1], M0isLTM1, cv::CMP_LT);
    cv::compare(M_split[0], M_split[2], M0isLTM2, cv::CMP_LT);
    cv::compare(M_split[1], M_split[2], M1isLTM2, cv::CMP_LT);


    cv::Mat M0isMaximum = (~M0isLTM1.getMat(cv::ACCESS_READ) & ~M0isLTM2.getMat(cv::ACCESS_READ))/255.0;
    M0isMaximum.convertTo(M0isMaximum, CV_32F);
    cv::Mat M1isMaximum = (M0isLTM1.getMat(cv::ACCESS_READ) & ~M1isLTM2.getMat(cv::ACCESS_READ))/255.0;
    M1isMaximum.convertTo(M1isMaximum, CV_32F);
    cv::Mat M2isMaximum = (M0isLTM2.getMat(cv::ACCESS_READ) & M1isLTM2.getMat(cv::ACCESS_READ))/255.0;
    M2isMaximum.convertTo(M2isMaximum, CV_32F);
    
    //cv::UMat M0isMaximumUM, M1isMaximumUM , M2isMaximumUM;

    //M0isMaximum.getUMat(cv::ACCESS_READ).copyTo(M0isMaximumUM);
    //M1isMaximum.getUMat(cv::ACCESS_READ).copyTo(M1isMaximumUM);
    //M2isMaximum.getUMat(cv::ACCESS_READ).copyTo(M2isMaximumUM);

    //M0isMaximum = (M0isLTM1 & M0isLTM2);
    //~M0isLTM2 ==>  M0isLTM2.release();

    cv::UMat O0, O1, O2;
    cv::multiply(M2isMaximum.getUMat(cv::ACCESS_READ), O_split[2], O2);
    cv::multiply(M1isMaximum.getUMat(cv::ACCESS_READ), O_split[1], O1);
    cv::multiply(M0isMaximum.getUMat(cv::ACCESS_READ), O_split[0], O0);

    cv::add(O0, O1, channelsGradMag_1);
    cv::add(channelsGradMag_1, O2, channelsGradMag_1);
  }


  cv::UMat test;
  if (m_normRad != 0)
  {
    cv::UMat S = convTri(channelsGradMag_0, m_normRad);
    //channelsGradMag_0 = channelsGradMag_0 / (S + m_normConst);
    cv::add(S, m_normConst, S);
    cv::divide(channelsGradMag_0, S, channelsGradMag_0);
  }

  gMag.push_back(channelsGradMag_0);
  gMag.push_back(channelsGradMag_1);

#ifdef DEBUG
  cv::imshow("M", gMag[0]);
  cv::imshow("O", gMag[1]);
  cv::waitKey();
#endif

  return gMag;
}



std::vector<cv::Mat>
ChannelsExtractorGradMagOpenCL::extractFeatures
  (
  cv::Mat img
  )
{
  int nChannels = img.channels();
  if ((nChannels != 1) && (nChannels != 3))
  {
    throw std::domain_error("Only gray level or BGR (3 channels) images allowed");
  }
   
  cv::Mat img_float;
  img.convertTo(img_float, CV_32FC1); // important to have continuous memory in img_aux.ptr<float>
  cv::UMat IM = img_float.getUMat(cv::ACCESS_READ);
  std::vector<cv::UMat> channelsGradMagUMat;
  channelsGradMagUMat = extractFeatures(IM);

  std::vector<cv::Mat> channelsGradMag;
  channelsGradMag.push_back(channelsGradMagUMat[0].getMat(cv::ACCESS_READ));

  /*
  cv::Size orig_sz = img.size();
  std::vector<cv::Mat> channelsGradMag(2);
  

  cv::UMat channelsGradMag_0 = cv::UMat::zeros(orig_sz.height, orig_sz.width, CV_32FC1);
  cv::UMat channelsGradMag_1 = cv::UMat::zeros(orig_sz.height, orig_sz.width, CV_32FC1);


  if (nChannels == 1)
  {
    cv::Mat kernel_horiz = (cv::Mat_<float>(1,3) << 0.5, 0, -0.5);
    k_horiz = kernel_horiz.getUMat(cv::ACCESS_READ);

    cv::UMat IM = img_float.getUMat(cv::ACCESS_READ);
    auto start = std::chrono::system_clock::now(); 
    for(int i = 0; i < 1; i++){
        gradMagOpenCL(IM,channelsGradMag_0,channelsGradMag_1,false);
    }  
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<float,std::milli> duration = end - start;
    std::cout << duration.count() << " time in OpenCL" << std::endl;
    cv::Mat chn0 = channelsGradMag_0.getMat(cv::ACCESS_READ);
    cv::Mat chn1 = channelsGradMag_1.getMat(cv::ACCESS_READ);

    //channelsGradMag[0] = channelsGradMag_0.getMat(cv::ACCESS_READ);
    //channelsGradMag[1] = channelsGradMag_1.getMat(cv::ACCESS_READ);
  }
  else if (nChannels == 3)
  {
    cv::Mat image_split[3];
    split(img, image_split);

    std::vector<cv::UMat> M_split(3);
    std::vector<cv::UMat> O_split(3);
    for (int i=0; i<3; i++)
    {
      image_split[i].convertTo(image_split[i], CV_32FC1); // important to have continuous memory in img_aux.ptr<float>
      M_split[i] = cv::UMat::zeros(orig_sz.height, orig_sz.width, CV_32FC1); //cv::Mat::zeros(orig_sz.height, orig_sz.width, CV_32FC1);
      O_split[i] = cv::UMat::zeros(orig_sz.height, orig_sz.width, CV_32FC1); //cv::Mat::zeros(orig_sz.height, orig_sz.width, CV_32FC1);
      gradMagOpenCL(image_split[i].getUMat(cv::ACCESS_READ), M_split[i], O_split[i], false);
    }

    // Compute M matrix
    cv::UMat channelsGradMag_0;
    cv::max(M_split[0], M_split[1], channelsGradMag_0);
    cv::max(channelsGradMag_0, M_split[2], channelsGradMag_0);

    //cv::UMat channelsGradMag_0 = cv::max(cv::max(M_split[0], M_split[1]), M_split[2]);
    
    // Compute O matrix (on each pixel we put the corresponding value on O_split[i] where
    // the M_split[i] has the maximum value across i=0,1,2).
    //cv::UMat M0isLTM1 = (M_split[0] < M_split[1]);
    /*cv::Mat M0isLTM2 = (M_split[0] < M_split[2]);
    cv::Mat M1isLTM2 = (M_split[1] < M_split[2]);* / 

    cv::UMat M0isLTM1, M0isLTM2 , M1isLTM2;
    cv::compare(M_split[0], M_split[1], M0isLTM1, cv::CMP_LT);
    cv::compare(M_split[0], M_split[2], M0isLTM2, cv::CMP_LT);
    cv::compare(M_split[1], M_split[2], M1isLTM2, cv::CMP_LT);


    cv::Mat M0isMaximum = (~M0isLTM1.getMat(cv::ACCESS_READ) & ~M0isLTM2.getMat(cv::ACCESS_READ))/255.0;
    M0isMaximum.convertTo(M0isMaximum, CV_32F);
    cv::Mat M1isMaximum = (M0isLTM1.getMat(cv::ACCESS_READ) & ~M1isLTM2.getMat(cv::ACCESS_READ))/255.0;
    M1isMaximum.convertTo(M1isMaximum, CV_32F);
    cv::Mat M2isMaximum = (M0isLTM2.getMat(cv::ACCESS_READ) & M1isLTM2.getMat(cv::ACCESS_READ))/255.0;
    M2isMaximum.convertTo(M2isMaximum, CV_32F);
    
    //cv::UMat M0isMaximum, M1isMaximum , M2isMaximum;

    //M0isMaximum = (M0isLTM1 & M0isLTM2);
    //~M0isLTM2 ==>  M0isLTM2.release();

    cv::UMat O0, O1, O2;
    cv::multiply(M2isMaximum, O_split[2], O2);
    cv::multiply(M1isMaximum, O_split[1], O1);
    cv::multiply(M0isMaximum, O_split[0], O0);


    cv::UMat channelsGradMag_1;
    cv::add(O0, O1, channelsGradMag_1);
    cv::add(channelsGradMag_1, O2, channelsGradMag_1);

    cv::Mat chn0 = channelsGradMag_0.getMat(cv::ACCESS_READ);
    cv::Mat chn1 = channelsGradMag_1.getMat(cv::ACCESS_READ);

    chn0.copyTo(channelsGradMag[0]);
    chn1.copyTo(channelsGradMag[1]);
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
  */
  return channelsGradMag;
}




