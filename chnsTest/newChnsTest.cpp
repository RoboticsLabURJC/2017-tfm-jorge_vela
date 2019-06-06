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

void MyEllipse( Mat img, double angle, int w, int h )
{
    int thickness = 2;
    int lineType = 8;

    ellipse( img,
       Point( w/2.0, h/2.0 ),
       Size( w/4.0, h/16.0 ),
       angle,
       0,
       360,
       Scalar( 255, 0, 0 ),
       thickness,
       lineType );
}


void MyFilledCircle( Mat img, Point center, int w )
{
    int thickness = -1;
    int lineType = 8;

    circle( img,
        center,
        w/32.0,
        Scalar( 0, 0, 255 ),
        thickness,
        lineType );
}

int main(int argc, const char* argv[])
{


	const int h=128, w=128  , misalign=1; int x, y, d; //192   const int h=12, w=12,
  	/*float* I=imageRead.ptr<float>(0), *I0=I+misalign; 
  	cv::Mat dummy_query = cv::Mat(128, 128, CV_8UC1, I0);*/
	Mat atom_image = Mat::zeros( h, w, CV_8UC1 );
	MyEllipse( atom_image, 90 , w, h);
	MyEllipse( atom_image, 0 , w, h);

	//MyEllipse( atom_image, 45 , width, height);
	//MyEllipse( atom_image, -45 , width, height);
	float *I=atom_image.ptr<float>(0), *I0=I+misalign; 
	cv::Mat dummy_query = cv::Mat(128, 128, CV_8UC1, I0);
	cv::imshow( "Display window", atom_image );
	cv::waitKey(0); 

  	//float I[h*w*3+misalign], *I0=I+misalign;


 	//for( x=0; x<h*w*3; x++ ) I0[x]=0;
 	//for( d=0; d<3; d++ ) I0[int(h*w/2+h/2)+d*h*w]=1;

	const int pad=2, rad=2, sf=sizeof(float); d=1;
  	const int h1=h+2*pad, w1=w+2*pad, h2=h1*2, w2=w1*2, h3=h2/4, w3=w2/4;
  	float *I1, *I2, *I3, *I4, *Gx, *Gy, *M, *O, *H, *G;

  	I1 = (float*) wrCalloc(h1*w1*d+misalign,sf) + misalign;

  	// perform tests of imPad, rgbConvert, convConst, resample and gradient
  	imPad(I0,I1,h,w,d,pad,pad,pad,pad,0,(float)CV_8UC1);
  	//printf("%lu \n", sizeof(I1));
  	dummy_query = cv::Mat(h1, w1, CV_8UC1, I1);
  	cv::imshow( "Display window 2", dummy_query);
	cv::waitKey(0);

 	/*I2 = rgbConvert(I1,h1*w1,d,0,1.0f);
  	dummy_query = cv::Mat(h1, w1, CV_8UC1, I2);
  	cv::imshow( "Display window 2", dummy_query);
	cv::waitKey(0);*/


	/*I3 = (float*) wrCalloc(h1*w1*d+misalign,sf) + misalign;
  	convTri(I2,I3,h1,w1,d,rad,1); 
  	dummy_query = cv::Mat(h1, w1, CV_8UC1, I3);
  	cv::imshow( "Display window 2", dummy_query);
	cv::waitKey(0);*/

	/*I4 = (float*) wrCalloc(h2*w2*d+misalign,sf) + misalign;
	resample(I0,I4,h1,h2,w1,w2,d,1.0f);
  	dummy_query = cv::Mat(height*2, width*2, CV_8UC1, I4);
  	cv::imshow( "Display window 2", dummy_query);
	cv::waitKey(0);*/

	return 0;
}