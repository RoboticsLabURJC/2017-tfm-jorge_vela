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

//#undef USE_SEPARABLE_CONVOLUTION
#define USE_SEPARABLE_CONVOLUTION

//#define DEBUG

ChannelsExtractorGradHistOpenCV::ChannelsExtractorGradHistOpenCV
  (
    int binSize,
    int nOrients,
    int softBin,
    int full
  ): ChannelsExtractorGradHist(binSize,
                               nOrients,
                               softBin,
                               full)
{
  createKernel(m_binSize,
               m_nOrients,
               m_softBin,
               m_full);
}

void
ChannelsExtractorGradHistOpenCV::createKernel
  (
  int bin,
  int nOrients,
  int softBin,
  bool full
  )
{
  // We use convolution with a special kernel to perform a weighted sum over a window of
  // bin x bin pixels using also the sum over bin x bin upper, upper left and left windows.
  // We are doing more computation than needed in the histogram. We sum over more windows
  // that are overlaping. We should be doing convolution with stride = bin to do it on not
  // overlapping windows. On the other hand, filter2D do not support stride.
  float xb, yb, xd, yd;
  int xb0, yb0;
  float sInv = 1.0/static_cast<float>(bin);
  float init = (0+.5f)*sInv - 0.5f;

  int klength = 2*bin;
  m_kernel = cv::Mat::zeros(klength, klength, CV_32F);

  xb = init + sInv*(bin/2);
  for (int x=0 ; x < klength; x++)
  {
    xb0 = (int)xb;
    xd = xb - xb0;
    xb += sInv;

    yb = init + sInv*(bin/2);
    for (int y = 0; y < klength; y++)
    {
      yb0 = (int)yb;
      yd = yb - yb0;
      yb += sInv;

      if ((y < bin) && (x < bin)) // 0 -> ms[3]
      {
        m_kernel.at<float>(y, x) = xd*yd;
      }
      else if ((y < bin) && (x >= bin)) // 2 -> ms[1]
      {
        m_kernel.at<float>(y, x) = yd - xd*yd;
      }
      else if ((y >= bin) && (x < bin)) // 1 -> ms[2]
      {
        m_kernel.at<float>(y, x) = xd - xd*yd;
      }
      else // if ((y >= bin) && (x >= bin)) // 3 -> ms[0]
      {
        m_kernel.at<float>(y, x) = 1.0 - xd - yd + xd*yd;
      }
    }
  }

  int kcenter = bin;
  if (bin % 2 == 1)
  {
    m_kernel = m_kernel(cv::Range(1,klength), cv::Range(1,klength));
    kcenter -= 1;
  }

//  kernel.copyTo(m_kernel);

#ifdef USE_SEPARABLE_CONVOLUTION
  // The kernel is separable so we get the 1d kernel to speed up convolution using SVD
  cv::SVD svd;
  svd(m_kernel);
  m_kernel1d = svd.u(cv::Range::all(), cv::Range(0,1)).clone(); // get first column of U as kernel 1d
  m_kernel1d *= sqrt(svd.w.at<float>(0,0)); // Muliply by the square root of the corresponding singular value.
#endif
}

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

  cv::Mat O0, M0, O0_eq_i;
  cv::Mat O1, M1, O1_eq_i;

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
#ifdef DEBUG
    std::cout << "=======> 1111111 " << std::endl;
#endif
    // no interpolation w.r.t. either orientation or spatial bin
    cv::Mat Haux;

    // First we obtain te images with gradient magnitude of the pixels
    // with a given quantized orientation and 0 in the rest of pixels.
    for (int i=0; i<nOrients; i++)
    {
      O0_eq_i = (O0 == i)/255; // Matrix with values 0 (false) and 1 (true)
      cv::multiply(O0_eq_i, M0, M0_orient_i, 1.0, CV_32F);

      // We use convolution with a full of ones kernel to sum over a window of
      // bin x bin pixels. We are doing more computation than needed as the
      // histogram, do not need overlaping windows and we should be doing convolution with
      // stride = bin. On the other hand, filter2D do not support stride.
      // For speed, we perform computation with two kernels as the box filter is separable.
      cv::Mat kernel = cv::Mat::ones(bin, 1, CV_32F);
      cv::filter2D(M0_orient_i, Haux, CV_32F, kernel, cv::Point(0, 0), 0, cv::BORDER_CONSTANT);
      cv::filter2D(Haux, Haux, CV_32F, kernel.t(), cv::Point(0, 0), 0, cv::BORDER_CONSTANT);

      // Use nearest neighbour interpolation to keep every bin rows and cols from Haux
      // (keep the right values of the Haux matrix).
      cv::resize(Haux, H[i], H[i].size(), 0, 0, cv::INTER_NEAREST);
    }
  }
  else if ( (softBin % 2 == 0) || (bin == 1) )
  {
#ifdef DEBUG
    std::cout << "=======> 2222222 " << std::endl;
#endif

    // interpolate w.r.t. orientation only, not spatial bin
    cv::Mat Haux0, Haux1, Hi1;

    // First we obtain te images with gradient magnitude of the pixels
    // with a given quantized orientation and 0 in the rest of pixels.
    for (int i=0; i<nOrients; i++)
    {
      O0_eq_i = (O0 == i)/255; // Matrix with values 0.0 (false) and 1.0 (true)
      cv::multiply(O0_eq_i, M0, M0_orient_i, 1.0, CV_32F);
      O1_eq_i = (O1 == i)/255; // Matrix with values 0.0 (false) and 1.0 (true)
      cv::multiply(O1_eq_i, M1, M1_orient_i, 1.0, CV_32F);

      // We use convolution with a full of ones kernel to sum over a window of
      // bin x bin pixels. We are doing more computation than needed as the
      // histogram, do not need overlaping windows and we should be doing convolution with
      // stride = bin. On the other hand, filter2D do not support stride.
      // For speed, we perform computation with two kernels as the box filter is separable.
      cv::Mat kernel = cv::Mat::ones(bin, 1, CV_32F);
      cv::filter2D(M0_orient_i, Haux0, CV_32F, kernel, cv::Point(0, 0), 0, cv::BORDER_CONSTANT);
      cv::filter2D(Haux0, Haux0, CV_32F, kernel.t(), cv::Point(0, 0), 0, cv::BORDER_CONSTANT);
      cv::filter2D(M1_orient_i, Haux1, CV_32F, kernel, cv::Point(0, 0), 0, cv::BORDER_CONSTANT);
      cv::filter2D(Haux1, Haux1, CV_32F, kernel.t(), cv::Point(0, 0), 0, cv::BORDER_CONSTANT);

      // Use nearest neighbour interpolation to keep every bin rows and cols from Haux
      // (keep the right values of the Haux matrix).
      cv::resize(Haux0, H[i], H[i].size(), 0, 0, cv::INTER_NEAREST);
      cv::resize(Haux1, Hi1, H[i].size(), 0, 0, cv::INTER_NEAREST);
      H[i] += Hi1;
    }
  }
  else
  {
#ifdef DEBUG
    std::cout << "=======> 333333333 " << std::endl;
#endif

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
      cv::multiply(O0_eq_i, M0, M0_orient_i, 1.0, CV_32F);
      if (m_softBin >= 0)
      {
        O1_eq_i = (O1 == i)/255;
        cv::multiply(O1_eq_i, M1, M1_orient_i, 1.0, CV_32F);
      }

      int kcenter = bin;
      if (bin % 2 == 1)
      {
        kcenter -= 1;
      }

#ifdef DEBUG
      if (i == 0)
      {
        std::cout << "ChannelsExtractorGradHistOpenCV::gradHist -->" << std::endl;
        std::cout << "=============================" << std::endl;
        std::cout << "wb, hb = " << wb << ", " << hb << std::endl;
        std::cout << "sInv2 = " << sInv2;
        std::cout << ", nOrients = " << nOrients;
        std::cout << ", full = " << full;
        std::cout << ", interpolate = " << (softBin>=0) << std::endl;

        std::cout << "O0 =" << std::endl;
        std::cout << O0 << std::endl;
        std::cout << "O0 == i ->" << std::endl;
        std::cout << (O0 == i) << std::endl;
        std::cout << "O0_eq_i = " << std::endl;
        std::cout << O0_eq_i << std::endl;
        std::cout << "M0_orient_i = " << std::endl;
        std::cout << M0_orient_i << std::endl;
      }
#endif

#ifdef USE_SEPARABLE_CONVOLUTION
      // The kernel is separable so we get the 1d kernel to speed up convolution using SVD
      cv::filter2D(M0_orient_i, Haux0, CV_32F, m_kernel1d, cv::Point(0, kcenter), 0, cv::BORDER_CONSTANT);
      cv::filter2D(Haux0, Haux0, CV_32F, m_kernel1d.t(), cv::Point(kcenter, 0), 0, cv::BORDER_CONSTANT);
      if (m_softBin >= 0)
      {
        cv::filter2D(M1_orient_i, Haux1, CV_32F, m_kernel1d, cv::Point(0, kcenter), 0, cv::BORDER_CONSTANT);
        cv::filter2D(Haux1, Haux1, CV_32F, m_kernel1d.t(), cv::Point(kcenter, 0), 0, cv::BORDER_CONSTANT);
      }
#else
      cv::filter2D(M0_orient_i, Haux0, CV_32F, m_kernel, cv::Point(kcenter,kcenter), 0, cv::BORDER_CONSTANT);
      if (m_softBin >= 0)
      {
        cv::filter2D(M1_orient_i, Haux1, CV_32F, m_kernel, cv::Point(kcenter,kcenter), 0, cv::BORDER_CONSTANT);
      }
#endif

#ifdef DEBUG
      if (i == 0)
      {
        std::cout << "Haux0 + Haux1 = " << std::endl;
        std::cout << Haux0 + Haux1 << std::endl;
      }
#endif

      // Use nearest neighbour interpolation to keep every bin rows and cols from Haux
      // (keep the right values of the Haux matrix).
      // The P.Dollar implementeation:
      //   - skips the first bin/2 columns and bin/2 rows althought they are added to the
      //     orientation bin that is to the right (in the first two columns) or down (in the first two rows).
      //   - We depart from the filtered images (H0 and H1) with the weighted sum of gradient magnitude on
      //     each quantized orienation in 2bin x 2bin regions. The sums should be performed in a non-overlapping
      //     fashiong starting at the bin/2 pixel (both in rows and columns). We use cv::resize with cv::INTER_NEAREST
      //     to select the sums corresponding to non-overlaping regions. In order to do as in P.Dollar's implementation
      //     we need to remove the last rows and columns in the image that makes the size not divisible by bin (are in
      //     outside any of the wb or hb bins). That is the reason to substract Haux0.cols%bin and Haux0.rows%bin to
      //     new_cols and new_rows, respectively.

      // shift the Haux0 matrix bin/2 pixels up and to the left.
      int new_rows = Haux0.rows - Haux0.rows % bin;
      int new_cols = Haux0.cols - Haux0.cols % bin;
      cv::Size sz_ajusted(new_cols, new_rows);
      cv::Mat out = cv::Mat::zeros(sz_ajusted, Haux0.type());
      int shift_x = bin/2.0;
      int shift_y = bin/2.0;
      int width_copy = Haux0.cols - shift_x - Haux0.cols % bin;
      int height_copy = Haux0.rows - shift_y - Haux0.rows % bin;
      cv::Rect copyToRect(0, 0, Haux0.cols - shift_x - Haux0.cols % bin, Haux0.rows - shift_y - Haux0.rows % bin);
      Haux0(cv::Rect(shift_x, shift_y, width_copy, height_copy)).copyTo(out(copyToRect));
      Haux0 = out;

      // shift the Haux1 matrix bin/2 pixels up and to the left.
      cv::Mat out2 = cv::Mat::zeros(sz_ajusted, Haux1.type());
      width_copy = Haux1.cols - shift_x - Haux1.cols % bin;
      height_copy = Haux1.rows - shift_y - Haux1.rows % bin;
      cv::Rect copyToRect2(0, 0, width_copy, height_copy);
      Haux1(cv::Rect(shift_x, shift_y, width_copy, height_copy)).copyTo(out2(copyToRect2));
      Haux1 = out2;

      cv::resize(Haux0, H[i], H[i].size(), 0, 0, cv::INTER_NEAREST);
      if (m_softBin >= 0)
      {
        cv::Mat Hi1;
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
  cv::Mat o_minusHalf;
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
    o_minusHalf = o - 0.5; // to make actualy a floor operation istead of a round one in convertTO CV_32S
    o_minusHalf.convertTo(O0, CV_32S); // Convert to int. OpenCV uses rounding but, as we have substracted 0.5, then it performs truntacion.
    O0.convertTo(O0_float, CV_32F); // Back to float
    od = o - O0_float;
    // O0 computation:
    O0.setTo(0, O0 >= nOrients);
    // O1 computation:
    O1 = O0 + 1;
    O1.setTo(0, O1 == nOrients);
    // M1 computation:
    m = M*norm;
    cv::multiply(m, od, M1);
    // M0 computation:
    M0 = m - M1;
  }
  else
  {
    // O0 computation:
    // In the P.Dollar implementation it is added 0.5 to o in order to convert with rounding (using the fact that truncation
    // is performed from float to int). OpenCV conversion from CV_32F to CV_32S is already performing rounding.
    o = O*oMult;
    o.convertTo(O0, CV_32S); // Convert to int (with rounding).
    O0.setTo(0, O0 >= nOrients);
    // M1 computation:
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


