/** ------------------------------------------------------------------------
 *
 *  @brief Implementation of Channel feature extractors for histogram gradients
 *  @author Jorge Vela
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2019/07/08
 *
 *  ------------------------------------------------------------------------ */

#include <iostream>
#include <channels/ChannelsExtractorGradHistOpenCV.h>
#include <opencv2/opencv.hpp>
#include "sse.hpp"


#define PI 3.14159265f

void
ChannelsExtractorGradHistOpenCV::gradHist
  (
  cv::Mat M,
  cv::Mat O,
  std::vector<cv::Mat>& H,
  int bin,
  int nOrients,
  int softBin,
  bool full
  )
{
  cv::Size sz = M.size();
  const int hb = sz.height/bin; // Number of bins in height
  const int wb = sz.width/bin;  // Number of bins in width
  const float s = (float)bin;   // number of pixels per bin in float
//  const float sInv = 1/s;
  const float sInv2 = 1/s/s;

  cv::Mat O0; // = new int[sz.height*sizeof(int)+16]();
  cv::Mat M0; // = new float[sz.height*sizeof(int)+16]();
  cv::Mat O1; // = new int[sz.height*sizeof(int)+16]();
  cv::Mat M1; // = new float[sz.height*sizeof(int)+16]();
  cv::Mat O0_eq_i;
  cv::Mat O1_eq_i;

  // Quantize the orientations into the nOrients bins.
  gradQuantize(O, M, sInv2, nOrients, full, softBin>=0, O0, O1, M0, M1);

  // Actually compute the nOrients histogram images. In this case there are
  // bin x bin squares where each pixels adds the magnitude of the gradient
  // to the image of the corresponding quantized gradient orientation index.

  // The P0[i] has the gradient magnitudes of all pixels with orientation i and
  // 0's if the orientation is different from i.
  std::vector<cv::Mat> P0(nOrients);
  std::vector<cv::Mat> P1(nOrients);

  if ( (softBin < 0) && (softBin % 2 == 0) )
  {
    // no interpolation w.r.t. either orientation or spatial bin
    cv::Mat Haux;

    // First we obtain te images with gradient magnitude of the pixels
    // with a given quantized orientation and 0 in the rest of pixels.
    for (int i=0; i<nOrients; i++)
    {
      O0_eq_i = (O0 == i)/255;
      O0_eq_i.convertTo(O0_eq_i, CV_32F);
      cv::multiply(O0_eq_i, M0, P0[i]);

      // We use convolution with a full of ones kernel to sum over a window of
      // bin x bin pixels. We are doing more computation than needed as the
      // histogram, do not need overlaping windows and we should be doing convolution with
      // stride = bin. On the other hand, filter2D do not support stride.
      cv::Mat kernel = cv::Mat::ones(bin, 1, CV_32F);
      cv::filter2D(P0[i], Haux, CV_32F, kernel, cv::Point(0,0), 0, cv::BORDER_CONSTANT);
      cv::filter2D(Haux, Haux, CV_32F, kernel.t(), cv::Point(0,0), 0, cv::BORDER_CONSTANT);

      // Use nearest neighbour interpolation to keep every bin rows and cols from Haux
      // (keep the right values of the Haux matrix).
      cv::resize(Haux, H[i], cv::Size(H[i].cols, H[i].rows), 0, 0, cv::INTER_NEAREST);
    }
  }
  else if ( (softBin % 2 == 0) || (bin == 1) )
  {
    // interpolate w.r.t. orientation only, not spatial bin
    cv::Mat Haux0;
    cv::Mat Haux1;
    cv::Mat Hi1;

    // First we obtain te images with gradient magnitude of the pixels
    // with a given quantized orientation and 0 in the rest of pixels.
    for (int i=0; i<nOrients; i++)
    {
      O0_eq_i = (O0 == i)/255;
      O0_eq_i.convertTo(O0_eq_i, CV_32F);
      O1_eq_i = (O1 == i)/255;
      O1_eq_i.convertTo(O1_eq_i, CV_32F);
      cv::multiply(O0_eq_i, M0, P0[i]);
      cv::multiply(O1_eq_i, M1, P1[i]);

      // We use convolution with a full of ones kernel to sum over a window of
      // bin x bin pixels. We are doing more computation than needed as the
      // histogram, do not need overlaping windows and we should be doing convolution with
      // stride = bin. On the other hand, filter2D do not support stride.
      cv::Mat kernel = cv::Mat::ones(bin, 1, CV_32F);
      cv::filter2D(P0[i], Haux0, CV_32F, kernel, cv::Point(0,0), 0, cv::BORDER_CONSTANT);
      cv::filter2D(Haux0, Haux0, CV_32F, kernel.t(), cv::Point(0,0), 0, cv::BORDER_CONSTANT);
      cv::filter2D(P1[i], Haux1, CV_32F, kernel, cv::Point(0,0), 0, cv::BORDER_CONSTANT);
      cv::filter2D(Haux1, Haux1, CV_32F, kernel.t(), cv::Point(0,0), 0, cv::BORDER_CONSTANT);

      // Use nearest neighbour interpolation to keep every bin rows and cols from Haux
      // (keep the right values of the Haux matrix).
      cv::resize(Haux0, H[i], cv::Size(H[i].cols, H[i].rows), 0, 0, cv::INTER_NEAREST);
      cv::resize(Haux1, Hi1, cv::Size(H[i].cols, H[i].rows), 0, 0, cv::INTER_NEAREST);
      H[i] += Hi1;
    }
  }
  else
  {
    //------------------------------------------------------------------------------
    // interpolate using trilinear interpolation:
    //   bilinear spatially, linear in the orientation

    // First we obtain te images with gradient magnitude of the pixels
    // with a given quantized orientation and 0 in the rest of pixels.
    cv::Mat Haux0;
    cv::Mat Haux1;

    for (int i=0; i<nOrients; i++)
    {
      O0_eq_i = (O0 == i)/255;
      O0_eq_i.convertTo(O0_eq_i, CV_32F);
      O1_eq_i = (O1 == i)/255;
      O1_eq_i.convertTo(O1_eq_i, CV_32F);
      cv::multiply(O0_eq_i, M0, P0[i]);
      cv::multiply(O1_eq_i, M1, P1[i]);

      // We use convolution with a full of ones kernel to sum over a window of
      // bin x bin pixels. We are doing more computation than needed as the
      // histogram, do not need overlaping windows and we should be doing convolution with
      // stride = bin. On the other hand, filter2D do not support stride.
#define USE_FILTER_1

#ifdef USE_FILTER_1
      int klength = 2*bin;
      cv::Mat kernel = cv::Mat::zeros(klength, 1, CV_32F);
      for (int k=0; k < klength; k++)
      {
         kernel.at<float>(k, 0) = klength - k;
      }
      kernel /= klength;
//      std::cout << "kernel = " << kernel << std::endl;

      cv::filter2D(P0[i], Haux0, CV_32F, kernel, cv::Point(0,0), 0, cv::BORDER_CONSTANT);
      cv::filter2D(Haux0, Haux0, CV_32F, kernel.t(), cv::Point(0,0), 0, cv::BORDER_CONSTANT);
      if (m_softBin >= 0)
      {
        cv::filter2D(P1[i], Haux1, CV_32F, kernel, cv::Point(0,0), 0, cv::BORDER_CONSTANT);
        cv::filter2D(Haux1, Haux1, CV_32F, kernel.t(), cv::Point(0,0), 0, cv::BORDER_CONSTANT);
      }
#endif


#ifdef USE_FILTER_3
      int klength = 2*bin;
      cv::Mat kernel = cv::Mat::zeros(klength, klength, CV_32F);
      float sInv = 1.0/bin;
      float init = sInv/bin;
      for (int j=0; j < klength; j++)
      {
        float j_norm;
        if (j==0)
        {
          j_norm = init;
        }
        else
        {
          j_norm += sInv; //static_cast<float>(j+1)/static_cast<float>(bin);
        }

        if (j == bin)
        {
          j_norm = init;
        }
        for (int k=0; k < klength; k++)
        {
          float k_norm;
          if (k==0)
          {
            k_norm = init;
          }
          else
          {
            k_norm += sInv; //static_cast<float>(j+1)/static_cast<float>(bin);
          }

          if (k == bin)
          {
            k_norm = init;
          }

//          ms[0]=1-xd-yd+xyd; ms[2]=xd-xyd;
//          ms[1]=yd-xyd;      ms[3]=xyd;

          if ((k < bin) && (j < bin)) // 0
          {
            kernel.at<float>(k, j) = k_norm*j_norm; //xyd; //1.0 - k_norm - j_norm + k_norm*j_norm;
          }
          else if ((k < bin) && (j >= bin)) // 2
          {
            kernel.at<float>(k, j) = k_norm - k_norm*j_norm; //yd-xyd; //k_norm - k_norm*j_norm;
          }
          else if ((k >= bin) && (j < bin)) // 1
          {
            kernel.at<float>(k, j) = j_norm - k_norm*j_norm; //xd-xyd; //j_norm - k_norm*j_norm;
          }
          else // if ((k >= bin) && (j >= bin)) // 3
          {
            kernel.at<float>(k, j) = 1.0 - k_norm - j_norm + k_norm*j_norm; //1-xd-yd+xyd; //1.0 - k_norm + k_norm*j_norm - j_norm;
          }
        }
      }
//      kernel = (cv::Mat_<float>(klength, klength) << 0.0625, 0.1875, 0.1875, 0.0625,
//                                                     0.1875, 0.5625, 0.5625, 0.1875,
//                                                     0.0625, 0.1875, 0.5625, 0.1875,
//                                                     0.0625, 0.1875, 0.1875, 0.0625);
      std::cout << "kernel = " << std::endl << kernel << std::endl;

      cv::filter2D(P0[i], Haux0, CV_32F, kernel, cv::Point(klength-1,klength-1), 0, cv::BORDER_CONSTANT);
      if (m_softBin >= 0)
      {
        cv::filter2D(P1[i], Haux1, CV_32F, kernel, cv::Point(klength-1,klength-1), 0, cv::BORDER_CONSTANT);
      }
#endif

      // Use nearest neighbour interpolation to keep every bin rows and cols from Haux
      // (keep the right values of the Haux matrix).
      cv::resize(Haux0, H[i], cv::Size(H[i].cols, H[i].rows), 0, 0, cv::INTER_NEAREST);
      if (m_softBin >= 0)
      {
        cv::Mat Hi1;
        cv::resize(Haux1, Hi1, cv::Size(H[i].cols, H[i].rows), 0, 0, cv::INTER_NEAREST);
        H[i] += Hi1;
      }
    }
  }

//  // normalize boundary bins which only get 7/8 of weight of interior bins
//  if ( softBin%2!=0 )
//  {
//    for( int o=0; o<nOrients; o++ )
//    {
//      // first row
//      H[o](cv::Range(0,1), cv::Range(0,wb)) *= 8.f/7.f;
//      // first column
//      H[o](cv::Range(0,hb), cv::Range(0,1)) *= 8.f/7.f;
//      // last row
//      H[o](cv::Range(hb-1,hb), cv::Range(0,wb)) *= 8.f/7.f;
//      // last column
//      H[o](cv::Range(0,hb), cv::Range(wb-1,wb)) *= 8.f/7.f;
//    }
//  }
}

void
ChannelsExtractorGradHistOpenCV::gradQuantize
  (
  cv::Mat O,
  cv::Mat M,
  float norm,
  int nOrients,
  bool full,
  bool interpolate,
  cv::Mat& O0,
  cv::Mat& O1,
  cv::Mat& M0,
  cv::Mat& M1
  )
{
  cv::Mat o;
  cv::Mat m;
  cv::Mat od;
  cv::Mat o0;
  cv::Mat O0_float;

  // define useful constants
  const float oMult = static_cast<float>(nOrients/(full?2*M_PI:M_PI));
  const int oMax = nOrients; // Total number of elements in all histograms

  // compute trailing locations without sse
  if ( interpolate )
  {
    o = O*oMult;
    o.convertTo(O0, CV_32S); // Convert to int (with truncation).
    O0.convertTo(O0_float, CV_32F); // Back to float
    od = o - O0_float;
    // O0
    O0.setTo(0, O0 >= oMax);
    // O1
    O1 = O0 + 1;
    O1.setTo(0, O1 >= oMax);
    // M1
    m = M*norm;
    cv::multiply(m, od, M1);
    // M0
    M0 = m - M1;
  }
  else
  {
    // O0
    o = O*oMult + 0.5;
    o.convertTo(O0, CV_32S); // Convert to int (with truncation).
    O0.setTo(0, O0 >= oMax);
    // M1
    M0 = M*norm;
    M1 = cv::Mat::zeros(M0.rows, M0.cols, CV_32F);
    O1 = cv::Mat::zeros(M0.rows, M0.cols, CV_32F);
  }
}


std::vector<cv::Mat>
ChannelsExtractorGradHistOpenCV::extractFeatures
  (
  cv::Mat img,
  std::vector<cv::Mat> gradMag
  )
{
  int h = img.size().height;
  int w = img.size().width;
  int hConv = h/m_binSize;
  int wConv = w/m_binSize;

  std::vector<cv::Mat> Hs(m_nOrients);
  for (int i=0; i < m_nOrients; i++)
  {
    Hs[i] = cv::Mat::zeros(hConv, wConv, CV_32FC1);
  }

  gradHist(gradMag[0],
           gradMag[1],
           Hs,
           m_binSize,
           m_nOrients,
           m_softBin,
           m_full);

  return Hs;
}


