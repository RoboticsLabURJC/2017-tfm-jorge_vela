/*******************************************************************************
* Piotr's Computer Vision Matlab Toolbox      Version 3.00
* Copyright 2014 Piotr Dollar.  [pdollar-at-gmail.com]
* Licensed under the Simplified BSD License [see external/bsd.txt]
*******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include "rgbConvertMex.cpp"
#include "imPadMex.cpp"
#include "convConst.cpp"
#include "imResampleMex.cpp"
#include "gradientMex.cpp"

#include <opencv2/highgui.hpp>
#include "opencv2/opencv.hpp"
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <iomanip>


#include <math.h>
#include <typeinfo>


using namespace cv;
using namespace std;

void print(Mat mat, int prec)
{     
    printf("%d %d \n", mat.size().height , mat.size().width );

    for(int i=0; i<mat.size().height; i++)
    {
        cout << "[";
        for(int j=0; j<mat.size().width; j++)
        {
            printf("%d",mat.at<int>(i,j));
            cout << setprecision(prec) << mat.at<double>(i,j);
            if(j != mat.size().width-1)
                cout << ", ";
            else
                cout << "]" << endl; 
        }
    }
}

// compile and test standalone channels source code
int main(int argc, const char* argv[])
{
  cv::Mat imageRead;
  //imageRead = cv::imread("sample.jpg" ,  CV_LOAD_IMAGE_COLOR);
  imageRead = cv::imread("sample.jpg" , CV_LOAD_IMAGE_GRAYSCALE );

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  int width = imageRead.size().width;
  int height = imageRead.size().height;

  printf("%i %i \n", width, height);


  float* buffer=imageRead.ptr<float>(0);  
  cv::Mat dummy_query = cv::Mat(128, 128, CV_8UC1, buffer);


  cv::imshow( "Display window", dummy_query );
  cv::waitKey(0); 

  int msalig=1;
  int d_test = 1;
  int sizefloat = sizeof(float);
  int height1=height;
  int width1=width ;
  float* convT = (float*) wrCalloc(height1*width1*d_test+msalig,sizefloat) + msalig;

  resample(buffer,convT,height1,height1,width1,width1,d_test,1.0f);
  cv::Mat dummy_query_2 = cv::Mat(128,128, CV_8UC1, convT);

  cv::imshow( "Display window", dummy_query_2);
  cv::waitKey(0); 

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /*int pad_test = 2;
  float *imgPad = (float*) wrCalloc(height1*width1*d_test+msalig,sizefloat) + msalig;
  imPad(buffer,imgPad,height1,width1,d_test,pad_test,pad_test,pad_test,pad_test,0,0.0f);


  dummy_query_2 = cv::Mat(130,130 , CV_8UC1, imgPad);

  cv::imshow( "Display window", dummy_query_2);
  cv::waitKey(0);*/ 
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // initialize test array (misalign controls memory mis-alignment)
  const int h=256, w=256  , misalign=1; int x, y, d; //192   const int h=12, w=12,
  float I[h*w*3+misalign], *I0=I+misalign;
  for( x=0; x<h*w*3; x++ ) I0[x]=0;
  for( d=0; d<3; d++ ) I0[int(h*w/2+h/2)+d*h*w]=1;

  // initialize memory for results with given misalignment
  const int pad=2, rad=2, sf=sizeof(float); d=3;
  const int h1=h+2*pad, w1=w+2*pad, h2=h1/2, w2=w1/2, h3=h2/4, w3=w2/4;
  float *I1, *I2, *I3, *I4, *Gx, *Gy, *M, *O, *H, *G;
  I1 = (float*) wrCalloc(h1*w1*d+misalign,sf) + misalign;
  I3 = (float*) wrCalloc(h1*w1*d+misalign,sf) + misalign;
  I4 = (float*) wrCalloc(h2*w2*d+misalign,sf) + misalign;
  Gx = (float*) wrCalloc(h2*w2*d+misalign,sf) + misalign;
  Gy = (float*) wrCalloc(h2*w2*d+misalign,sf) + misalign;
  M  = (float*) wrCalloc(h2*w2*d+misalign,sf) + misalign;
  O  = (float*) wrCalloc(h2*w2*d+misalign,sf) + misalign;
  H  = (float*) wrCalloc(h3*w3*d*6+misalign,sf) + misalign;
  G  = (float*) wrCalloc(h3*w3*d*24+misalign,sf) + misalign;
 


  // perform tests of imPad, rgbConvert, convConst, resample and gradient
  imPad(I0,I1,h,w,d,pad,pad,pad,pad,0,0.0f);

  //Entra una imagen RGB y sale una imagen en otro espacio de color segun el parametro flag 
  //0 = gray, 1 = luv, 2 = hsv. 
  I2 = rgbConvert(I1,h1*w1,d,0,1.0f); d=1;
  //cv::Mat imageReadGray;

  // Convolucion rapida de la imagen utilizando un filtro triangular. Util para 
  // realizar el suavizado de la imagen. 
  convTri(I2,I3,h1,w1,d,rad,1); 
  
  // Remuestrea utilizando interpolacion bilinear. 
  resample(I3,I4,h1,h2,w1,w2,d,1.0f);
  
  // Calcula los gradientes de x e y en cada punto.
  // Imagen de entrada = I4, Imagen salida Gx, Gy.
  grad2( I4, Gx, Gy, h2, w2, d );
  
  // Calcula el gradiente de la magnitud y de la orientacion en cada
  // punto.
  gradMag( I4, M, O, h2, w2, d , true); //add true
  
  // Calcula los histogramas de gradiente por bin de bloques de pixeles.
  gradHist(M,O,H,h2,w2,4,6,0, true); //add true
 
  // Calcula las características HOG
  hog(M,O,H,h2,w2,20,4,6,true, .2f);
  //hog(H,G,h2,w2,4,6,.2f);
  

  /*
  // print some test arrays
  printf("---------------- M: ----------------\n");
  for(y=0;y<h2;y++){ for(x=0;x<w2;x++) printf("%.4f ",M[x*h2+y]); printf("\n");}
  printf("---------------- O: ----------------\n");
  for(y=0;y<h2;y++){ for(x=0;x<w2;x++) printf("%.4f ",O[x*h2+y]); printf("\n");}
  */

  /*cv::Mat image = cv::Mat(h, w, CV_8UC4, (unsigned*)O);
  cv::imshow( "Display window", image );
  cv::waitKey(0);*/
  // free memory and return
  wrFree(I1-misalign); wrFree(I2); wrFree(I3-misalign); wrFree(I4-misalign);
  wrFree(Gx-misalign); wrFree(Gy-misalign); wrFree(M-misalign);
  wrFree(O-misalign); wrFree(H-misalign); wrFree(G-misalign);
  system("pause");
  return 0;
}
