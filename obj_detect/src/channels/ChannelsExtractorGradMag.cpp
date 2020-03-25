/** ------------------------------------------------------------------------
 *
 *  @brief Implementation of Channel feature extractors for LUV color space.
 *  @author Jorge Vela
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2019/07/08
 *
 *  ------------------------------------------------------------------------ */

//ChanExtractorMex.cpp
#include <iostream>
#include <channels/ChannelsExtractorGradMag.h>
#include <opencv/cv.hpp>

#include "sse.hpp"

using namespace cv;
using namespace std;


#define PI 3.14159265f

// compute x and y gradients for just one column (uses sse)
void grad1( float *I, float *Gx, float *Gy, int h, int w, int x ) {
  int y, y1; float *Ip, *In, r; __m128 *_Ip, *_In, *_G, _r;
  // compute column of Gx
  Ip=I-h; In=I+h; r=.5f;
  if(x==0) { r=1; Ip+=h; } else if(x==w-1) { r=1; In-=h; }
  if( h<4 || h%4>0 || (size_t(I)&15) || (size_t(Gx)&15) ) {
    for( y=0; y<h; y++ ) *Gx++=(*In++-*Ip++)*r;
  } else {
    _G=(__m128*) Gx; _Ip=(__m128*) Ip; _In=(__m128*) In; _r = SET(r);
    for(y=0; y<h; y+=4) *_G++=MUL(SUB(*_In++,*_Ip++),_r);
  }
  // compute column of Gy
  #define GRADY(r) *Gy++=(*In++-*Ip++)*r;
  Ip=I; In=Ip+1;
  // GRADY(1); Ip--; for(y=1; y<h-1; y++) GRADY(.5f); In--; GRADY(1);
  y1=((~((size_t) Gy) + 1) & 15)/4; if(y1==0) y1=4; if(y1>h-1) y1=h-1;
  GRADY(1); Ip--; for(y=1; y<y1; y++) GRADY(.5f);
  _r = SET(.5f); _G=(__m128*) Gy;
  for(; y+4<h-1; y+=4, Ip+=4, In+=4, Gy+=4)
    *_G++=MUL(SUB(LDu(*In),LDu(*Ip)),_r);
  for(; y<h-1; y++) GRADY(.5f); In--; GRADY(1);
  #undef GRADY
}

// compute x and y gradients at each location (uses sse)
void grad2( float *I, float *Gx, float *Gy, int h, int w, int d ) {
  int o, x, c, a=w*h; for(c=0; c<d; c++) for(x=0; x<w; x++) {
    o=c*a+x*h; grad1( I+o, Gx+o, Gy+o, h, w, x );
  }
}

// build lookup table a[] s.t. a[x*n]~=acos(x) for x in [-1,1]
float* acosTable() {
  const int n=10000, b=10; int i;
  static float a[n*2+b*2]; static bool init=false;
  float *a1=a+n+b; if( init ) return a1;
  for( i=-n-b; i<-n; i++ )   a1[i]=PI;
  for( i=-n; i<n; i++ )      a1[i]=float(acos(i/float(n)));
  for( i=n; i<n+b; i++ )     a1[i]=0;
  for( i=-n-b; i<n/10; i++ ) if( a1[i] > PI-1e-6f ) a1[i]=PI-1e-6f;
  init=true; return a1;
}


void GradMagExtractor::gradMagNorm( float *M, float *S, int h, int w, float norm ) {
  __m128 *_M, *_S, _norm; int i=0, n=h*w, n4=n/4;
  _S = (__m128*) S; _M = (__m128*) M; _norm = SET(norm);
  bool sse = !(size_t(M)&15) && !(size_t(S)&15);
  if(sse) for(; i<n4; i++) { *_M=MUL(*_M,RCP(ADD(*_S++,_norm))); _M++; }
  if(sse) i*=4; for(; i<n; i++) M[i] /= (S[i] + norm);
}


// compute gradient magnitude and orientation at each location (uses sse)
void gradMag( float *I, float *M, float *O, int h, int w, int d, bool full ) {
  int x, y, y1, c, h4, s; float *Gx, *Gy, *M2; __m128 *_Gx, *_Gy, *_M2, _m;
  float *acost = acosTable(), acMult=10000.0f;
  // allocate memory for storing one column of output (padded so h4%4==0)
  h4=(h%4==0) ? h : h-(h%4)+4; s=d*h4*sizeof(float);

  M2= new float[s+16](); _M2=(__m128*) M2;
  Gx= new float[s+16](); _Gx=(__m128*) Gx;
  Gy= new float[s+16](); _Gy=(__m128*) Gy;


  //M2=(float*) malloc(s+16); _M2=(__m128*) M2;
  //Gx=(float*) malloc(s+16); _Gx=(__m128*) Gx;
  //Gy=(float*) malloc(s+16); _Gy=(__m128*) Gy;
  // compute gradient magnitude and orientation for each column
  for( x=0; x<w; x++ ) {
    // compute gradients (Gx, Gy) with maximum squared magnitude (M2)
    for(c=0; c<d; c++) {
      grad1( I+x*h+c*w*h, Gx+c*h4, Gy+c*h4, h, w, x );

      //grad1( I+x*h+c*w*h, Gx+c*h4, Gy+c*h4, h, w, x );
      for( y=0; y<h4/4; y++ ) {
        y1=h4/4*c+y;
        _M2[y1]=ADD(MUL(_Gx[y1],_Gx[y1]),MUL(_Gy[y1],_Gy[y1]));
        if( c==0 ) {continue;} _m = CMPGT( _M2[y1], _M2[y] );
        _M2[y] = OR( AND(_m,_M2[y1]), ANDNOT(_m,_M2[y]) );
        _Gx[y] = OR( AND(_m,_Gx[y1]), ANDNOT(_m,_Gx[y]) );
        _Gy[y] = OR( AND(_m,_Gy[y1]), ANDNOT(_m,_Gy[y]) );
      }
    }
    // compute gradient mangitude (M) and normalize Gx
    for( y=0; y<h4/4; y++ ) {
      _m = MIN( RCPSQRT(_M2[y]), SET(1e10f) );
      _M2[y] = RCP(_m);
      if(O) _Gx[y] = MUL( MUL(_Gx[y],_m), SET(acMult) );
      if(O) _Gx[y] = XOR( _Gx[y], AND(_Gy[y], SET(-0.f)) );
    };
    memcpy( M+x*h, M2, h*sizeof(float) );
    // compute and store gradient orientation (O) via table lookup

    if( O!=0 ) for( y=0; y<h; y++ ) O[x*h+y] = acost[(int)Gx[y]]; //error en esta linea ??

    if( O!=0 && full ) {
      y1=((~size_t(O+x*h)+1)&15)/4; y=0;
      for( ; y<y1; y++ ) O[y+x*h]+=(Gy[y]<0)*PI;
      for( ; y<h-4; y+=4 ) STRu( O[y+x*h],
        ADD( LDu(O[y+x*h]), AND(CMPLT(LDu(Gy[y]),SET(0.f)),SET(PI)) ) );
      for( ; y<h; y++ ) O[y+x*h]+=(Gy[y]<0)*PI;
    }
  }
  free(Gx); free(Gy); free(M2);
}


 float* GradMagExtractor::allocW(int size , int sf, int misalign){
   float *var;
   var  = (float*) calloc(size+misalign,sf) + misalign;
   return var;
 }


 void GradMagExtractor::gradM(float *I, float *M, float *O){
	const int h=12, w=12  , misalign=1; int x, y, d=3; 
  	gradMag( I, M, O, h, w, d ,  false ); 
 }	

 void GradMagExtractor::gradMAdv(cv::Mat image, float *M, float *O){
    int h = image.size().height;
    int w = image.size().width;
    int nChannels = image.channels();

    int size = h*w*nChannels;
    int sizeData = sizeof(float);
    int misalign=1;

    //float I[h*w*nChannels+misalign], *I0=I+misalign;
    //for(int x=0; x<h*w*nChannels; x++ ) I0[x]=0;

    if(nChannels == 1){
        cv::Mat dst;
        image.convertTo(dst, CV_32F);
        transpose(dst, dst);
        dst = dst/255.0;
        float *data = dst.ptr<float>();

        gradMag(data, M, O, h, w, nChannels,  false ); 

    }else if(nChannels == 3){
        float *M1 = new float[size](); 
        float *O1 = new float[size]();

        float *M2 = new float[size](); 
        float *O2 = new float[size]();

        float *M3 = new float[size](); 
        float *O3 = new float[size]();

        cv::Mat image_split[3];
        split(image,image_split);

        cv::Mat dst;
        image_split[0].convertTo(dst, CV_32F);
        transpose(dst, dst);
        dst = dst/255.0;
        float *data = dst.ptr<float>();
        gradMag(data, M1, O1, h, w, 1,  false ); 

        cv::Mat dst2;
        image_split[1].convertTo(dst2, CV_32F);
        transpose(dst2, dst2);
        dst2 = dst2/255.0;
        float *data2 = dst2.ptr<float>();
        gradMag(data2, M2, O2, h, w, 1,  false ); 


        cv::Mat dst3;
        image_split[2].convertTo(dst3, CV_32F);
        transpose(dst3, dst3);
        dst3 = dst3/255.0;
        float *data3 = dst3.ptr<float>();
        gradMag(data3, M3, O3, h, w, 1,  false ); 

        int tot = h*w;
        for(int i=0; i < tot; i++){

          float Mfinal = 0;
          if(M1[i] > Mfinal){
            Mfinal = M1[i];
          }

          if(M2[i] > Mfinal){
            Mfinal = M2[i];
          }

          if(M3[i] > Mfinal){
            Mfinal = M3[i];
          }
          M[i] = Mfinal;
        }

        for(int i=0; i < tot; i++){

          float Ofinal = 999999;
          if(O1[i] < Ofinal){
            Ofinal = O1[i];
          }

          if(O2[i] < Ofinal){
            Ofinal = O2[i];
          }

          if(O3[i] < Ofinal){
            Ofinal = O3[i];
          }
          O[i] = Ofinal;
        }      

        //printf("%.4f %.4f %.4f\n", O1[0], O2[0], O[0]);
    }
    
 }  