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



/*
void
gradHist
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
  const int h0 = hb*bin;        // number of pixels in height
  const int w0 = wb*bin;        // number of pixels in width
  const int nb = wb*hb;         // #bins height x #bins width
  const float s = (float)bin;   // number of pixels per bin in float
  const float sInv = 1/s;
  const float sInv2 = 1/s/s;
//  float *H0, *H1, *M0, *M1;
//  int x, y;
//  int *O0, *O1;
//  float xb, init;

  O0 = new int[sz.height*sizeof(int)+16]();
  M0 = new float[sz.height*sizeof(int)+16]();
  O1 = new int[sz.height*sizeof(int)+16]();
  M1 = new float[sz.height*sizeof(int)+16]();

  // main loop
  for( x=0; x<w0; x++ )
  {
    // compute target orientation bins for entire column - very fast
    gradQuantize(O+x*h,M+x*h,O0,O1,M0,M1,nb,h0,sInv2,nOrients,full,softBin>=0);
    if ( softBin<0 && softBin%2==0 )
    {
      // no interpolation w.r.t. either orientation or spatial bin
      H1=H+(x/bin)*hb;
      #define GH H1[O0[y]]+=M0[y]; y++;
      if( bin==1 )      for(y=0; y<h0;) { GH; H1++; }
      else if( bin==2 ) for(y=0; y<h0;) { GH; GH; H1++; }
      else if( bin==3 ) for(y=0; y<h0;) { GH; GH; GH; H1++; }
      else if( bin==4 ) for(y=0; y<h0;) { GH; GH; GH; GH; H1++; }
      else for( y=0; y<h0;) { for( int y1=0; y1<bin; y1++ ) { GH; } H1++; }
      #undef GH

    }
    else if ( softBin%2==0 || bin==1 )
    {
      // interpolate w.r.t. orientation only, not spatial bin
      H1=H+(x/bin)*hb;
      #define GH H1[O0[y]]+=M0[y]; H1[O1[y]]+=M1[y]; y++;
      if( bin==1 )      for(y=0; y<h0;) { GH; H1++; }
      else if( bin==2 ) for(y=0; y<h0;) { GH; GH; H1++; }
      else if( bin==3 ) for(y=0; y<h0;) { GH; GH; GH; H1++; }
      else if( bin==4 ) for(y=0; y<h0;) { GH; GH; GH; GH; H1++; }
      else for( y=0; y<h0;) { for( int y1=0; y1<bin; y1++ ) { GH; } H1++; }
      #undef GH
    }
    else
    {
      //------------------------------------------------------------------------------
      // interpolate using trilinear interpolation
      float ms[4], xyd, yb, xd, yd; __m128 _m, _m0, _m1;
      bool hasLf, hasRt; int xb0, yb0;
      if( x==0 ) { init=(0+.5f)*sInv-0.5f; xb=init; }
      hasLf = xb>=0; xb0 = hasLf?(int)xb:-1; hasRt = xb0 < wb-1;
      xd=xb-xb0; xb+=sInv; yb=init; y=0;

      // macros for code conciseness
      #define GHinit yd=yb-yb0; yb+=sInv; H0=H+xb0*hb+yb0; xyd=xd*yd; \
        ms[0]=1-xd-yd+xyd; ms[1]=yd-xyd; ms[2]=xd-xyd; ms[3]=xyd;
      #define GH(H,ma,mb) H1=H; STRu(*H1,ADD(LDu(*H1),MUL(ma,mb)));

      // leading rows, no top bin
      for( ; y<bin/2; y++ ) {
        yb0=-1; GHinit;
        if(hasLf) { H0[O0[y]+1]+=ms[1]*M0[y]; H0[O1[y]+1]+=ms[1]*M1[y]; }
        if(hasRt) { H0[O0[y]+hb+1]+=ms[3]*M0[y]; H0[O1[y]+hb+1]+=ms[3]*M1[y]; }
      }

      // main rows, has top and bottom bins, use SSE for minor speedup
      if( softBin<0 ) for( ; ; y++ ) {
        yb0 = (int) yb; if(yb0>=hb-1) break; GHinit; _m0=SET(M0[y]);
        if(hasLf) { _m=SET(0,0,ms[1],ms[0]); GH(H0+O0[y],_m,_m0); }
        if(hasRt) { _m=SET(0,0,ms[3],ms[2]); GH(H0+O0[y]+hb,_m,_m0); }
      } else for( ; ; y++ ) {
        yb0 = (int) yb; if(yb0>=hb-1) break; GHinit;
        _m0=SET(M0[y]); _m1=SET(M1[y]);
        if(hasLf) { _m=SET(0,0,ms[1],ms[0]);
          GH(H0+O0[y],_m,_m0); GH(H0+O1[y],_m,_m1); }
        if(hasRt) { _m=SET(0,0,ms[3],ms[2]);
          GH(H0+O0[y]+hb,_m,_m0); GH(H0+O1[y]+hb,_m,_m1); }
      }

      // final rows, no bottom bin
      for( ; y<h0; y++ ) {
        yb0 = (int) yb; GHinit;
        if(hasLf) { H0[O0[y]]+=ms[0]*M0[y]; H0[O1[y]]+=ms[0]*M1[y]; }
        if(hasRt) { H0[O0[y]+hb]+=ms[2]*M0[y]; H0[O1[y]+hb]+=ms[2]*M1[y]; }
      }
      #undef GHinit
      #undef GH
    }
  }

  delete[] O0;
  delete[] O1;
  delete [] M0;
  delete[] M1;

  // normalize boundary bins which only get 7/8 of weight of interior bins
  if ( softBin%2!=0 )
  {
    for( int o=0; o<nOrients; o++ )
    {
      x=0; for( y=0; y<hb; y++ ) H[o*nb+x*hb+y]*=8.f/7.f;
      y=0; for( x=0; x<wb; x++ ) H[o*nb+x*hb+y]*=8.f/7.f;
      x=wb-1; for( y=0; y<hb; y++ ) H[o*nb+x*hb+y]*=8.f/7.f;
      y=hb-1; for( x=0; x<wb; x++ ) H[o*nb+x*hb+y]*=8.f/7.f;
    }
  }
}
*/

void
ChannelsExtractorGradHistOpenCV::gradQuantize
  (
  cv::Mat O,
  cv::Mat M,
  int nb,
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
  const int oMax = nOrients*nb; // Total number of elements in all histograms

  // compute trailing locations without sse
  if ( interpolate )
  {
    o = O*oMult;
    o.convertTo(O0, CV_32S); // Convert to int (trunc).
    O0.convertTo(O0_float, CV_32F); // Back to float
    od = o - O0_float;
    // O0
    O0 *= nb;
    O0.setTo(0, O0 >= oMax);
    // O1
    O1 = O0 + nb;
    O1.setTo(0.0, O1 == oMax);
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
    o.convertTo(O0, CV_32S); // Convert to int (trunc).
    O0 *= nb;
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
  int dChan = img.channels();

  int hConv = h/m_binSize;
  int wConv = w/m_binSize;

  std::vector<cv::Mat> Hs(m_nOrients);
  for (int i=0; i < m_nOrients; i++)
  {
    Hs[i] = cv::Mat::zeros(hConv, wConv, CV_32FC1);
  }

//  gradHistOpenCV(M, O, Hs, m_binSize, m_nOrients, m_softBin, m_full);

//  for(int i = 0; i < m_nOrients; i++)
//  {
//    transpose(Hs[i], Hs[i]);
//  }

  return Hs;
}


