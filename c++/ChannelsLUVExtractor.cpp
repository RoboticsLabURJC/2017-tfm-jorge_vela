#include "ChannelsLUVExtractor.h"
#include <iostream>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>

using namespace std;


std::vector<cv::Mat> ChannelsLUVExtractor::extractFeatures(cv::Mat img){
	std::vector<cv::Mat> imagesLUV(3);
	cv::Mat imageLUV;
	cvtColor(img, imageLUV, CV_BGR2XYZ); 
	cv::Mat luv[3];
	imshow( "Image Tot " , imageLUV );
	split(imageLUV,luv);  
	cv::Mat L = luv[0];
	cv::Mat U = luv[1];
	cv::Mat V = luv[2];
	imagesLUV[0] = L;
	imagesLUV[1] = U;
	imagesLUV[2] = V;
	return imagesLUV;
}
	

/*MÉTODO 1 PARA OBTENER LUV ???
Mat bgr[3];   //destination array
split(image,bgr);//split source  
Mat X = 0.412453*bgr[2] + 0.35758 *bgr[1] + 0.180423*bgr[0];
Mat Y = 0.212671*bgr[2] + 0.71516 *bgr[1] + 0.072169*bgr[0];
Mat Z = 0.019334*bgr[2] + 0.119193*bgr[1] + 0.950227*bgr[0];*/
