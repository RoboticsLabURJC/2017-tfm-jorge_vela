/** ------------------------------------------------------------------------
 *
 *  @brief Implementation of Channel feature extractors for magnitude and orient gradients.
 *  @author Jorge Vela
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2019/07/08
 *
 *  ------------------------------------------------------------------------ */

#include <iostream>
#include <channels/ChannelsExtractorGradMag.h>
#include <channels/Utils.h>
#include <opencv2/opencv.hpp>
#include "sse.hpp"

#define PI 3.14159265f

// compute x and y gradients for just one column (uses sse)
/**
 * Funcion grad1. Calcula los gradientes x e y por cada columna
 *
 */
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
/**
 * Calcula los gradientes x e y en cada píxel
 * 
 *
 */
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

// normalize gradient magnitude at each location (uses sse)
/**
 * Funcion gradMagNorm. Normaliza la magnitud del gradiente en cada píxel.
 *
 * @param M: Dirección de memoria donde está la magnitud del gradiente a normalizar
 * @param S: Nueva dirección de memoria donde se guarda la magnitud del gradiente normalizada
 * @param h: Altura correspondiente al cv::Mat de la magnitud del gradiente
 * @param w: Anchura correspondiente al cv::Mat de la magnitud del gradiente
 * @param norm: Valor por el que se quiere normalizar
 *
 */
void GradMagExtractor::gradMagNorm( float *M, float *S, int h, int w, float norm ) {
  __m128 *_M, *_S, _norm; int i=0, n=h*w, n4=n/4;
  _S = (__m128*) S; _M = (__m128*) M; _norm = SET(norm);
  bool sse = !(size_t(M)&15) && !(size_t(S)&15);
  if(sse) for(; i<n4; i++) { *_M=MUL(*_M,RCP(ADD(*_S++,_norm))); _M++; }
  if(sse) i*=4; for(; i<n; i++) M[i] /= (S[i] + norm);
}


// compute gradient magnitude and orientation at each location (uses sse)
/**
 * Funcion gradMag. Se ha obtenido de p.dollar y alguna cosa modificada.
 * Calcula la magnitud y orientación del gradiente en cada pixel. 
 *
 * @param I: Imagen de la cual se quieren calcular las características
 * @param M: Dirección de memoria donde se guardara la magnitud del gradiente
 * @param O: Direccion de memoria donde se guardara la orientación del gradiente
 * @param h: Altura (píxeles) de la imagen
 * @param w: Anchura (píxeles) de la imagen
 * @param d: Número de canales de la imagen
 * @param full: Si es verdadero calcula ángulos en [0,2*pi), sino en [0,pi)
 *
 */
void gradMag( float *I, float *M, float *O, int h, int w, int d, bool full ) {
  int x, y, y1, c, h4, s; float *Gx, *Gy, *M2; __m128 *_Gx, *_Gy, *_M2, _m;
  float *acost = acosTable(), acMult=10000.0f;
  // allocate memory for storing one column of output (padded so h4%4==0)
  h4=(h%4==0) ? h : h-(h%4)+4; s=d*h4*sizeof(float);

  M2= new float[s+16](); _M2=(__m128*) M2;
  Gx= new float[s+16](); _Gx=(__m128*) Gx;
  Gy= new float[s+16](); _Gy=(__m128*) Gy;

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

    if( O!=0 ) for( y=0; y<h; y++ ) O[x*h+y] = acost[(int)Gx[y]]; 

    if( O!=0 && full ) {
      y1=((~size_t(O+x*h)+1)&15)/4; y=0;
      for( ; y<y1; y++ ) O[y+x*h]+=(Gy[y]<0)*PI;
      for( ; y<h-4; y+=4 ) STRu( O[y+x*h],
        ADD( LDu(O[y+x*h]), AND(CMPLT(LDu(Gy[y]),SET(0.f)),SET(PI)) ) );
      for( ; y<h; y++ ) O[y+x*h]+=(Gy[y]<0)*PI;
    }
  }
  delete(Gx); delete(Gy); delete(M2);     
}

/**
 * Funcion gradM. Tiene como entrada la imagen de la que se quieren obtener la magnitud y orientación
 * del gradiente y la variable donde se guardarán los valores. Retorna el vector de cv::Mat con las
 * características. La funcion extractFeatures llama a esta función.
 *
 * @param image: Imagen de la cual se quieren obtener las características.
 * @param *M: Dirección de memoria donde se guarda la magnitud del gradiente.
 * @param *O: Dirección de memoria donde se guarda la orientación del gradiente. 
 *
 * @return std::vector<cv::Mat>: Estructura que contiene las imagenes con los la magnitud y orientación.
 */
std::vector<cv::Mat> GradMagExtractor::gradM(cv::Mat image, float *M, float *O){
  int h = image.size().height;
  int w = image.size().width;
  int nChannels = image.channels();

  int size = h*w*nChannels;
  int sizeData = sizeof(float);
  int misalign=1;

  if(nChannels == 1)
  {
    cv::Mat dst;
    image.convertTo(dst, CV_32F);
    transpose(dst, dst);
    dst = dst/255.0;
    float *data = dst.ptr<float>();

    gradMag(data, M, O, h, w, nChannels,  false ); //ESTABA AL CONTRARIO ( w,h) ¿porque?

  }
  else if(nChannels == 3)
  {
    //std::vector<cv::Mat> channelsGradMag(2);

    std::vector<float*> MVal{new float[size](), new float[size](), new float[size]() };
    std::vector<float*> OVal{new float[size](), new float[size](), new float[size]() };

    transpose(image, image);  //cambio que se hacia por cada canal
    //image = image/255;        //cambio que se hacia por cada canal
    cv::Mat image_split[3];
    split(image,image_split); 

    cv::Mat dst;
    image_split[0].convertTo(dst, CV_32F);
    float *data = dst.ptr<float>();

    gradMag(data, MVal[0], OVal[0], h, w, 1,  false ); 

    cv::Mat dst2;
    image_split[1].convertTo(dst2, CV_32F);
    float *data2 = dst2.ptr<float>();
    gradMag(data2, MVal[1], OVal[1], h, w, 1,  false ); 

    cv::Mat dst3;
    image_split[2].convertTo(dst3, CV_32F);
    float *data3 = dst3.ptr<float>();
    gradMag(data3, MVal[2], OVal[2], h, w, 1,  false ); 


    int tot = h*w;

    for(int i=0; i < tot; i++)
    {
      int max = ( MVal[0][i] < MVal[1][i] ) ? 1  : 0 ;
      int max2 =  ( MVal[max][i] < MVal[2][i] ) ? 2 : max;

      M[i] = MVal[max2][i];
      O[i] = OVal[max2][i];
    }
  }  

  if(m_normRad != 0){
    cv::Mat dummy_query = cv::Mat(w, h, CV_32FC1, M);
    cv::Mat M_to_img = convTri(dummy_query, m_normRad);
    cv::Mat newM;
    M_to_img.convertTo(newM, CV_32FC1);    
    float *dataM = newM.ptr<float>();
    gradMagNorm(M, dataM, w,h, m_normConst);
  }

  std::vector<cv::Mat> channelsGradMag(2);
  cv::Mat gradM = cv::Mat(w,h, CV_32FC1, M);
  transpose(gradM, gradM);
  channelsGradMag[0] = gradM;
  cv::Mat gradO = cv::Mat(w,h, CV_32FC1, O);
  transpose(gradO, gradO);
  channelsGradMag[1] = gradO;

  return channelsGradMag;
}

/**
 * Función extractFeatures. 
 * Se le pasa una imagen y se encarga de calcular la magnitud del gradiente y la orientación del gradiente.
 * La orientación del graciente no la utiliza como parámetro final, pero sirve para calcular el histograma.
 *
 * @param img: Contiene la imagen de la cual se quieren obtener las características
 * @return std::vector<cv::Mat>: Vector con la magnitud y el gradiente en formato cv::Mat
 */
std::vector<cv::Mat> GradMagExtractor::extractFeatures(cv::Mat img){

  //printf("GradMag: -->8UC3 %d , 8UC1 %d , imageType %d \n",CV_8UC3, CV_8UC1 ,  img.type() );

  //assert(img.type() == CV_8UC3 || img.type() == CV_8UC1);


  int dChan = img.channels();
  int width = img.size().width;
  int height = img.size().height;

  int size = width*height*dChan;
  float *M = new float[size](); 
  float *O = new float[size]();  

  std::vector<cv::Mat> channelsGradMag(2);
  channelsGradMag = gradM(img,M,O);

  return channelsGradMag;
}  
