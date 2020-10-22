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
  const float s = static_cast<float>(bin);   // number of pixels per bin in float
  const float sInv2 = 1.0/s/s;

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
  cv::Mat M0_orient_i;
  cv::Mat M1_orient_i;

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
      cv::multiply(O0_eq_i, M0, M0_orient_i);

      // We use convolution with a full of ones kernel to sum over a window of
      // bin x bin pixels. We are doing more computation than needed as the
      // histogram, do not need overlaping windows and we should be doing convolution with
      // stride = bin. On the other hand, filter2D do not support stride.
      cv::Mat kernel = cv::Mat::ones(bin, 1, CV_32F);
      cv::filter2D(M0_orient_i, Haux, CV_32F, kernel, cv::Point(0, 0), 0, cv::BORDER_CONSTANT);
      cv::filter2D(Haux, Haux, CV_32F, kernel.t(), cv::Point(0, 0), 0, cv::BORDER_CONSTANT);

//      // shift the Haux0 matrix bin/2 pixels up and 1 pixel to the left.
//      cv::Mat out = cv::Mat::zeros(Haux.size(), Haux.type());
//      Haux(cv::Rect(bin/2, bin/2, Haux.cols-(bin/2), Haux.rows-(bin/2))).copyTo(out(cv::Rect(0, 0, Haux.cols-bin/2, Haux.rows-bin/2)));
//      Haux = out;

      // Use nearest neighbour interpolation to keep every bin rows and cols from Haux
      // (keep the right values of the Haux matrix).
      cv::resize(Haux, H[i], H[i].size(), 0, 0, cv::INTER_NEAREST);
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
      cv::multiply(O0_eq_i, M0, M0_orient_i);
      cv::multiply(O1_eq_i, M1, M1_orient_i);

      // We use convolution with a full of ones kernel to sum over a window of
      // bin x bin pixels. We are doing more computation than needed as the
      // histogram, do not need overlaping windows and we should be doing convolution with
      // stride = bin. On the other hand, filter2D do not support stride.
      cv::Mat kernel = cv::Mat::ones(bin, 1, CV_32F);
      cv::filter2D(M0_orient_i, Haux0, CV_32F, kernel, cv::Point(0, 0), 0, cv::BORDER_CONSTANT);
      cv::filter2D(Haux0, Haux0, CV_32F, kernel.t(), cv::Point(0, 0), 0, cv::BORDER_CONSTANT);
      cv::filter2D(M1_orient_i, Haux1, CV_32F, kernel, cv::Point(0, 0), 0, cv::BORDER_CONSTANT);
      cv::filter2D(Haux1, Haux1, CV_32F, kernel.t(), cv::Point(0, 0), 0, cv::BORDER_CONSTANT);

//      // shift the Haux0 matrix bin/2 pixels up and 1 pixel to the left.
//      cv::Mat out = cv::Mat::zeros(Haux0.size(), Haux0.type());
//      Haux0(cv::Rect(bin/2, bin/2, Haux0.cols-(bin/2), Haux0.rows-(bin/2))).copyTo(out(cv::Rect(0, 0, Haux0.cols-bin/2, Haux0.rows-bin/2)));
//      Haux0 = out;

//      // shift the Haux1 matrix bin/2 pixels up and 1 pixel to the left.
//      cv::Mat out2 = cv::Mat::zeros(Haux1.size(), Haux1.type());
//      Haux1(cv::Rect(bin/2, bin/2, Haux1.cols-(bin/2), Haux1.rows-(bin/2))).copyTo(out2(cv::Rect(0, 0, Haux1.cols-bin/2, Haux1.rows-bin/2)));
//      Haux1 = out2;

      // Use nearest neighbour interpolation to keep every bin rows and cols from Haux
      // (keep the right values of the Haux matrix).
      cv::resize(Haux0, H[i], H[i].size(), 0, 0, cv::INTER_NEAREST);
      cv::resize(Haux1, Hi1, H[i].size(), 0, 0, cv::INTER_NEAREST);
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
      cv::multiply(O0_eq_i, M0, M0_orient_i);
      cv::multiply(O1_eq_i, M1, M1_orient_i);

      // We use convolution with a special kernel to perform a weighted sum over a window of
      // bin x bin pixels using also the bin x bin windows upper, upper left and to the left.
      // We are doing more computation than needed as the histogram, do not need overlaping
      // windows and we should be doing convolution with stride = bin. On the other hand,
      // filter2D do not support stride.
#define USE_FILTER_3

#ifdef USE_FILTER_1
      int klength = 2*bin;
      cv::Mat kernel = cv::Mat::zeros(klength, 1, CV_32F);
      for (int k=0; k < klength; k++)
      {
         kernel.at<float>(k, 0) = klength - k;
//         kernel.at<float>(k, 0) = k+1;
      }
      kernel /= klength;
      std::cout << "kernel = " << kernel << std::endl;

      cv::filter2D(P0[i], Haux0, CV_32F, kernel, cv::Point(0,bin), 0, cv::BORDER_CONSTANT);
      cv::filter2D(Haux0, Haux0, CV_32F, kernel.t(), cv::Point(bin,0), 0, cv::BORDER_CONSTANT);
      if (m_softBin >= 0)
      {
        cv::filter2D(P1[i], Haux1, CV_32F, kernel, cv::Point(0,bin), 0, cv::BORDER_CONSTANT);
        cv::filter2D(Haux1, Haux1, CV_32F, kernel.t(), cv::Point(bin,0), 0, cv::BORDER_CONSTANT);
      }
#endif

#ifdef USE_FILTER_2
      int klength = 2*bin;
      cv::Mat kernel = cv::Mat::ones(klength, 1, CV_32F);
      std::cout << "kernel = " << kernel << std::endl;

      cv::filter2D(P0[i], Haux0, CV_32F, kernel, cv::Point(0,bin), 0, cv::BORDER_CONSTANT);
      cv::filter2D(Haux0, Haux0, CV_32F, kernel.t(), cv::Point(bin,0), 0, cv::BORDER_CONSTANT);
      if (m_softBin >= 0)
      {
        cv::filter2D(P1[i], Haux1, CV_32F, kernel, cv::Point(0,bin), 0, cv::BORDER_CONSTANT);
        cv::filter2D(Haux1, Haux1, CV_32F, kernel.t(), cv::Point(bin,0), 0, cv::BORDER_CONSTANT);
      }
#endif

#ifdef USE_FILTER_3
      float xb, yb, xd, yd;
      int xb0, yb0;
      float sInv = 1.0/bin;
      int bin2 = bin/2;
      float init = (0+.5f)*sInv - 0.5f + sInv*static_cast<float>(bin2);

      int klength = 2*bin;
      cv::Mat kernel = cv::Mat::zeros(klength, klength, CV_32F);

      xb = init;
      for (int x=0; x < klength; x++)
      {
        xb0 = (int)xb;
        xd = xb - xb0;
        xb += sInv;

        yb = init;
        for (int y=0; y < klength; y++)
        {
          yb0 = (int)yb;
          yd = yb - yb0;
          yb += sInv;

          if ((y < bin) && (x < bin)) // 0 -> ms[3]
          {
            kernel.at<float>(y, x) = xd*yd;
          }
          else if ((y < bin) && (x >= bin)) // 2 -> ms[1]
          {
            kernel.at<float>(y, x) = yd - xd*yd;
          }
          else if ((y >= bin) && (x < bin)) // 1 -> ms[2]
          {
            kernel.at<float>(y, x) = xd - xd*yd;
          }
          else // if ((y >= bin) && (x >= bin)) // 3 -> ms[0]
          {
            kernel.at<float>(y, x) = 1 - xd - yd + xd*yd;
          }
        }
      }
//      std::cout << "kernel = " << std::endl;
//      std::cout << kernel << std::endl;

//      // Set to 0 first and last bin/2 rows
//      M0_orient_i(cv::Range(0, bin/2), cv::Range::all()) = 0.0;
//      M0_orient_i(cv::Range(M0_orient_i.rows-bin/2, M0_orient_i.rows), cv::Range::all()) = 0.0;

//      // Set to 0 first and last bin/2 columns
//      M0_orient_i(cv::Range::all(), cv::Range(0, bin/2)) = 0.0;
//      M0_orient_i(cv::Range::all(), cv::Range(M0_orient_i.cols-bin/2, M0_orient_i.cols)) = 0.0;

//      // Set to 0 first and last bin/2 rows
//      M1_orient_i(cv::Range(0, bin/2), cv::Range::all()) = 0.0;
//      M1_orient_i(cv::Range(M0_orient_i.rows-bin/2, M0_orient_i.rows), cv::Range::all()) = 0.0;

//      // Set to 0 first and last bin/2 columns
//      M1_orient_i(cv::Range::all(), cv::Range(0, bin/2)) = 0.0;
//      M1_orient_i(cv::Range::all(), cv::Range(M1_orient_i.cols-bin/2, M1_orient_i.cols)) = 0.0;

      cv::filter2D(M0_orient_i, Haux0, CV_32F, kernel, cv::Point(bin,bin), 0, cv::BORDER_CONSTANT);
      if (m_softBin >= 0)
      {
        cv::filter2D(M1_orient_i, Haux1, CV_32F, kernel, cv::Point(bin,bin), 0, cv::BORDER_CONSTANT);
      }

//      if (i==0)
//      {
//        std::cout << "Haux0 = " << std::endl;
//        std::cout << Haux0 << std::endl;
////      std::cout << "Haux0.size() = " << Haux0.size() << std::endl;
////      std::cout << "H[i].size() = " << H[i].size() << std::endl;
//      }
#endif

      // Use nearest neighbour interpolation to keep every bin rows and cols from Haux
      // (keep the right values of the Haux matrix).

      // shift the Haux0 matrix bin/2 pixels up and 1 pixel to the left.
      cv::Mat out = cv::Mat::zeros(Haux0.size(), Haux0.type());
      Haux0(cv::Rect(bin/2, bin/2, Haux0.cols-(bin/2), Haux0.rows-(bin/2))).copyTo(out(cv::Rect(0, 0, Haux0.cols-bin/2, Haux0.rows-bin/2)));
      Haux0 = out;

      // shift the Haux1 matrix bin/2 pixels up and 1 pixel to the left.
      cv::Mat out2 = cv::Mat::zeros(Haux1.size(), Haux1.type());
      Haux1(cv::Rect(bin/2, bin/2, Haux1.cols-(bin/2), Haux1.rows-(bin/2))).copyTo(out2(cv::Rect(0, 0, Haux1.cols-bin/2, Haux1.rows-bin/2)));
      Haux1 = out2;

// //      cv::resize(Haux0(cv::Range((bin/2), Haux0.rows), cv::Range((bin/2), Haux0.cols)), H[i], H[i].size(), 0, 0, cv::INTER_NEAREST);
//      cv::resize(Haux0(cv::Rect(bin/2, bin/2, Haux1.cols-(bin/2), Haux1.rows-(bin/2))), H[i], H[i].size(), 0, 0, cv::INTER_NEAREST);
// //      cv::resize(Haux0(cv::Rect(bin/2, bin/2, Haux1.cols-bin, Haux1.rows-bin)), H[i], H[i].size(), 0, 0, cv::INTER_NEAREST);
      cv::resize(Haux0, H[i], H[i].size(), 0, 0, cv::INTER_NEAREST);
      if (m_softBin >= 0)
      {
        cv::Mat Hi1;
// //        cv::resize(Haux1(cv::Range((bin/2), Haux1.rows), cv::Range((bin/2), Haux1.cols)), Hi1, H[i].size(), 0, 0, cv::INTER_NEAREST);
//        cv::resize(Haux1(cv::Rect(bin/2, bin/2, Haux1.cols-(bin/2), Haux1.rows-(bin/2))), Hi1, H[i].size(), 0, 0, cv::INTER_NEAREST);
// //        cv::resize(Haux1(cv::Rect(bin/2, bin/2, Haux1.cols-bin, Haux1.rows-bin)), Hi1, H[i].size(), 0, 0, cv::INTER_NEAREST);
        cv::resize(Haux1, Hi1, H[i].size(), 0, 0, cv::INTER_NEAREST);
        H[i] += Hi1;
      }
    }
  }

  // normalize boundary bins which only get 7/8 of weight of interior bins
  if ( softBin%2!=0 )
  {
    for( int o=0; o<nOrients; o++ )
    {
      // first column
      H[o](cv::Range::all(), cv::Range(0,1)) *= 8.f/7.f;
      // first row
      H[o](cv::Range(0,1), cv::Range::all()) *= 8.f/7.f;
      // last column
      H[o](cv::Range::all(), cv::Range(wb-1,wb)) *= 8.f/7.f;
      // last row
      H[o](cv::Range(hb-1,hb), cv::Range::all()) *= 8.f/7.f;
    }
  }
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

  // compute trailing locations
  if ( interpolate )
  {
    o = O*oMult;
    o.convertTo(O0, CV_32S); // Convert to int (with truncation).
    O0.convertTo(O0_float, CV_32F); // Back to float
    od = o - O0_float;
    // O0
    O0.setTo(0, O0 >= nOrients);
    // O1
    O1 = O0 + 1;
    O1.setTo(0, O1 >= nOrients);
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
    O0.setTo(0, O0 >= nOrients);
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


