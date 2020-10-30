/** ------------------------------------------------------------------------
 *
 *  @brief Implementation of Channel feature extractors for histogram gradients
 *  @author Jorge Vela
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2019/07/08
 *
 *  ------------------------------------------------------------------------ */

#include <iostream>
#include <channels/ChannelsExtractorGradHistPDollar.h>
#include <opencv2/opencv.hpp>
#include "sse.hpp"


#define PI 3.14159265f

// helper for gradHist, quantize O and M into O0, O1 and M0, M1 (uses sse)
void
ChannelsExtractorGradHistPDollar::gradQuantize
  (
  float *O,
  float *M,
  int *O0,
  int *O1,
  float *M0,
  float *M1,
  int nb,
  int n,
  float norm,
  int nOrients,
  bool full,
  bool interpolate
  )
{
  // assumes all *OUTPUT* matrices are 4-byte aligned
  int i, o0, o1; float o, od, m;
  __m128i _o0, _o1, *_O0, *_O1; __m128 _o, _od, _m, *_M0, *_M1;

  // define useful constants
  const float oMult=(float)nOrients/(full?2*PI:PI);
  const int oMax=nOrients*nb; // Total number of elements in all histograms
  const __m128 _norm=SET(norm), _oMult=SET(oMult), _nbf=SET((float)nb);
  const __m128i _oMax=SET(oMax), _nb=SET(nb);

  // perform the majority of the work with sse
  _O0=(__m128i*) O0;
  _O1=(__m128i*) O1;
  _M0=(__m128*) M0;
  _M1=(__m128*) M1;
  if  ( interpolate )
  {
    for( i=0; i<=n-4; i+=4 )         // Do it for 4 values at a time with SSE
    {
      _o=MUL(LDu(O[i]),_oMult);      // o = O[i] * oMult
      _o0=CVT(_o);                   // o0 = floor(o) (truncation)
      _od=SUB(_o,CVT(_o0));          // od = o - o0 // Decimals in o0
      _o0=CVT(MUL(CVT(_o0),_nbf));   // o0 = floor(floor(o0) * nbf)
      _o0=AND(CMPGT(_oMax,_o0),_o0); // if (oMax <= o0) o0 = 0.0
      *_O0++=_o0;                    // O0[i] = o0
      _o1=ADD(_o0,_nb);              // o1 = o0 + nb
      _o1=AND(CMPGT(_oMax,_o1),_o1); // if (oMax <= o1) o1 = 0.0
      *_O1++=_o1;                    // O1[i] = o1
      _m=MUL(LDu(M[i]),_norm);       // m = M[i] * norm // norm is 1/bin_size
      *_M1=MUL(_od,_m);              // M1[i] = od * m
      *_M0++=SUB(_m,*_M1);           // M0[i] = m - M1[i]
      _M1++;
    }
  }
  else
  {
    for( i=0; i<=n-4; i+=4 )         // Do it for 4 values at a time with SSE
    {
      _o=MUL(LDu(O[i]),_oMult);      // o = O[i] * oMult
      _o0=CVT(ADD(_o,SET(.5f)));     // o0 = o + 0.5
      _o0=CVT(MUL(CVT(_o0),_nbf));   // o0 *= nbf
      _o0=AND(CMPGT(_oMax,_o0),_o0); // if(o0>=oMax) o0 = 0.0;
      *_O0++=_o0;                    // O0[i] = o0
      *_M0++=MUL(LDu(M[i]),_norm);   // M0[i] = M[i] * norm // norm is 1/bin_size
      *_M1++=SET(0.f);               // M1[i] = 0.0
      *_O1++=SET(0);                 // O1[i] = 0
    }
  }

  // compute trailing locations without sse
  if ( interpolate )
  {
    for(; i<n; i++ )
    {
      o=O[i]*oMult;
      o0=(int) o;
      od=o-o0;
      o0*=nb;
      if(o0>=oMax) o0=0;
      O0[i]=o0;
      o1=o0+nb;
      if(o1==oMax) o1=0;
      O1[i]=o1;
      m=M[i]*norm;
      M1[i]=od*m;
      M0[i]=m-M1[i];
    }
  }
  else
  {
    for(; i<n; i++ )
    {
      o=O[i]*oMult;
      o0=(int) (o+.5f);
      o0*=nb;
      if(o0>=oMax) o0=0;
      O0[i]=o0;
      M0[i]=M[i]*norm;
      M1[i]=0;
      O1[i]=0;
    }
  }
}

// compute nOrients gradient histograms per bin x bin block of pixels
/**
 *
 * Funcion gradHist: Calcula los histogramas del gradiente de nOrients
 * en bloques de binxbin pixeles
 *
 * @param M: Magnitud del gradiente
 * @paramr O: Orientacion del gradiente
 * @param H: Dirección de memoria donde se guarda el histograma del gradiente
 * @param h: Altura de la imagen de la que se calcula el histograma
 * @param w: Ancho de la imagen de la que se calcula el histograma
 * @param bin: Tamaño de los bloques (binxbin) de los que se calcula los pixeles
 * @param nOrients: Numero de orientaciónes de los bin.
 * @param softBin: Tamaño del suavizado de los bloques ??
 * @param full: Si es verdadero calcula ángulos en [0,2*pi), sino en [0,pi)
 *
 */
void
ChannelsExtractorGradHistPDollar::gradHist
  (
  float *M,
  float *O,
  float *H,
  int h,
  int w,
  int bin,
  int nOrients,
  int softBin,
  bool full
  )
{
  const int hb=h/bin, wb=w/bin, h0=hb*bin, w0=wb*bin, nb=wb*hb;
  const float s=(float)bin, sInv=1/s, sInv2=1/s/s;
  float *H0, *H1, *M0, *M1;
  int x, y;
  int *O0, *O1;
  float xb, init;

  O0 = new int[h*sizeof(int)+16]();
  M0 = new float[h*sizeof(int)+16]();
  O1 = new int[h*sizeof(int)+16]();
  M1 = new float[h*sizeof(int)+16]();

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
      float ms[4], xyd, yb, xd, yd;
      __m128 _m, _m0, _m1;
      bool hasLf, hasRt;
      int xb0, yb0;
      if( x==0 ) { init=(0+.5f)*sInv-0.5f; xb=init; }
      hasLf = xb>=0;
      xb0 = hasLf?(int)xb:-1;
      hasRt = xb0 < wb-1;
      xd=xb-xb0;
      xb+=sInv;
      yb=init;
      y=0;

      // macros for code conciseness
      #define GHinit yd=yb-yb0; yb+=sInv; H0=H+xb0*hb+yb0; xyd=xd*yd; \
        ms[0]=1-xd-yd+xyd; ms[1]=yd-xyd; ms[2]=xd-xyd; ms[3]=xyd;
      #define GH(H,ma,mb) H1=H; STRu(*H1,ADD(LDu(*H1),MUL(ma,mb)));

      // leading rows, no top bin
      for( ; y<bin/2; y++ )
      {
        yb0=-1;
        GHinit;
        if(hasLf)
        {
          H0[O0[y]+1]+=ms[1]*M0[y];
          H0[O1[y]+1]+=ms[1]*M1[y];
        }
        if(hasRt)
        {
          H0[O0[y]+hb+1]+=ms[3]*M0[y];
          H0[O1[y]+hb+1]+=ms[3]*M1[y];
        }
      }

      // main rows, has top and bottom bins, use SSE for minor speedup
      if ( softBin<0 )
      {
        for( ; ; y++ )
        {
          yb0 = (int) yb;
          if(yb0>=hb-1) break;
          GHinit;

          _m0=SET(M0[y]);
          if(hasLf)
          {
            _m=SET(0,0,ms[1],ms[0]);
            GH(H0+O0[y],_m,_m0);
          }
          if(hasRt)
          {
            _m=SET(0,0,ms[3],ms[2]);
            GH(H0+O0[y]+hb,_m,_m0);
          }
        }
      }
      else
      {
        for( ; ; y++ )
        {
          yb0 = (int) yb;
          if(yb0>=hb-1) break;
          GHinit;
          _m0=SET(M0[y]);
          _m1=SET(M1[y]);
          if(hasLf)
          {
            _m=SET(0,0,ms[1],ms[0]);
            GH(H0+O0[y],_m,_m0);
            GH(H0+O1[y],_m,_m1);
          }
          if(hasRt)
          {
            _m=SET(0,0,ms[3],ms[2]);
            GH(H0+O0[y]+hb,_m,_m0);
            GH(H0+O1[y]+hb,_m,_m1);
          }
        }
      }

      // final rows, no bottom bin
      for( ; y<h0; y++ )
      {
        yb0 = (int) yb;
        GHinit;
        if(hasLf)
        {
          H0[O0[y]]+=ms[0]*M0[y];
          H0[O1[y]]+=ms[0]*M1[y];
        }
        if(hasRt)
        {
          H0[O0[y]+hb]+=ms[2]*M0[y];
          H0[O1[y]+hb]+=ms[2]*M1[y];
        }
      }
      #undef GHinit
      #undef GH
    }
  }

  delete[] O0;
  delete[] O1;
  delete[] M0;
  delete[] M1;

  // normalize boundary bins which only get 7/8 of weight of interior bins
  if ( softBin % 2 != 0 )
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

/**
 * Funcion gradH. Encargada de calcular el histograma del gradiente, teniendo la imagen
 * y las direcciones de memoria con la magnitud y orientación del gradiente.
 *
 * @param image: Image de la cual se desea calcular el histograma
 * @param M: Magnitud del gradiente
 * @param O: Orientación del gradiente
 * @param H: Donde se guardan valores del histograma con los valores como float
 *
 * @retunr std::vector<cv::Mat>: Vector con los histogramas del gradiente como cv::Mat
 */
std::vector<cv::Mat>
ChannelsExtractorGradHistPDollar::gradH
  (
  cv::Mat image,
  float *M,
  float *O
  )
{
  int h = image.size().height;
  int w = image.size().width;
  int dChan = image.channels();

  int hConv = h/m_binSize;
  int wConv = w/m_binSize;
  int size = hConv*wConv*dChan*m_nOrients;

  float *H = new float[size]();
  gradHist(M, O, H, h, w, m_binSize, m_nOrients, m_softBin, m_full);

  // Create the cv::Mat images copying memory from the H buffer.
  std::vector<cv::Mat> H2;
  int nb = hConv*wConv;
  for(int i = 0; i < m_nOrients; i++)
  {
    //arr[i] = &H[i*pos];
    cv::Mat gradH = cv::Mat(wConv, hConv, CV_32FC1, &H[i*nb]).clone(); // <-- copy
    transpose(gradH, gradH);
    H2.push_back(gradH);
  }
  delete [] H; // As we have copyed the memory we can delete H.

  return H2;
}


/**
 * Función extractFeatures. 
 * Se le pasa una imagen, junto a la magnitud y orientación del gradiente y se encarga de calcular
 * el histograma del gradiente.
 *
 * @param img: Contiene la imagen de la cual se quieren obtener las características
 * @param gradMag: Vector con los cv::Mat correspondientes a la magnitud y orientacion del gradiente.
 * @return std::vector<cv::Mat>: Vector los distintos cv::Mat correspondientes a los histogramas del gradiente
 */
std::vector<cv::Mat>
ChannelsExtractorGradHistPDollar::extractFeatures
  (
  cv::Mat img,
  std::vector<cv::Mat> gradMag
  )
{
  transpose(gradMag[0], gradMag[0]);
  float *dataM = gradMag[0].ptr<float>();

  transpose(gradMag[1], gradMag[1]);
  float *dataO = gradMag[1].ptr<float>();

  std::vector<cv::Mat> channelsGradHist;
  channelsGradHist = gradH(img, dataM, dataO);

  return channelsGradHist;
}
