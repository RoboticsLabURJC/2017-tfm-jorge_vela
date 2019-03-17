#include "ChannelsBordersExtractor.h"

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>

using namespace std;

cv::Mat extractObliquePositive(cv::Mat imageGRAY){
	cv::Mat imageBlackGray;
	cv::Mat imageblack(imageGRAY.rows, imageGRAY.cols, CV_8UC3, cv::Scalar(0,0,0));
    cvtColor(imageblack, imageBlackGray, CV_BGR2GRAY);
	
	for(int i = 1; i < imageBlackGray.rows-1; i++) //[1 1 0; 1 0 -1; 0 -1 -1]
	{
		for(int j = 1; j < imageBlackGray.cols-1; j++)
		{
			int val1 = 1*(int)imageGRAY.at<uchar>(i-1,j-1);
			int val2 = 1*(int)imageGRAY.at<uchar>(i-1,j);
			int val3 = 0*(int)imageGRAY.at<uchar>(i-1,j+1);
			int val4 = 1*(int)imageGRAY.at<uchar>(i,j-1);
			int val5 = 0*(int)imageGRAY.at<uchar>(i,j);
			int val6 = -1*(int)imageGRAY.at<uchar>(i,j+1);
 			int val7 = 0*(int)imageGRAY.at<uchar>(i+1,j-1);
			int val8 = -1*(int)imageGRAY.at<uchar>(i+1,j);
			int val9 = -1*(int)imageGRAY.at<uchar>(i+1,j+1);
			
			int tot = val1+val2+val3+val4+val5+val6+val7+val8+val9;
			imageBlackGray.at<uchar>(i,j) = tot;
		}
	}
	return imageBlackGray;
}

cv::Mat extractObliqueNegative(cv::Mat imageGRAY){
	cv::Mat imageBlackGray;
	cv::Mat imageblack(imageGRAY.rows, imageGRAY.cols, CV_8UC3, cv::Scalar(0,0,0));
    cvtColor(imageblack, imageBlackGray, CV_BGR2GRAY);
	
	for(int i = 1; i < imageBlackGray.rows-1; i++) //[1 1 0; 1 0 -1; 0 -1 -1]
	{
		for(int j = 1; j < imageBlackGray.cols-1; j++)
		{
			int val1 = -1*(int)imageGRAY.at<uchar>(i-1,j-1);
			int val2 = -1*(int)imageGRAY.at<uchar>(i-1,j);
			int val3 = 0*(int)imageGRAY.at<uchar>(i-1,j+1);
			int val4 = -1*(int)imageGRAY.at<uchar>(i,j-1);
			int val5 = 0*(int)imageGRAY.at<uchar>(i,j);
			int val6 = 1*(int)imageGRAY.at<uchar>(i,j+1);
 			int val7 = 0*(int)imageGRAY.at<uchar>(i+1,j-1);
			int val8 = 1*(int)imageGRAY.at<uchar>(i+1,j);
			int val9 = 1*(int)imageGRAY.at<uchar>(i+1,j+1);
			
			int tot = val1+val2+val3+val4+val5+val6+val7+val8+val9;
			imageBlackGray.at<uchar>(i,j) = tot;
		}
	}
	return imageBlackGray;
}


std::vector<cv::Mat> ChannelsBordersExtractor::extractFeatures(cv::Mat image){
	std::vector<cv::Mat> imagesBorders(4);
	cv::Mat imageGRAY;
    cvtColor(image, imageGRAY, CV_BGR2GRAY);
	cv::Mat obliquePositive = extractObliquePositive(imageGRAY);
	cv::Mat obliqueNegative = extractObliqueNegative(image);
    
    cv::Mat grad_x, grad_y;
    int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;
	Sobel(imageGRAY, grad_x, ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT );
	Sobel(imageGRAY, grad_y, ddepth, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT );   
	
    imagesBorders[0] = obliquePositive;
    imagesBorders[1] = obliquePositive;
	imagesBorders[2] = grad_x;
	imagesBorders[3] = grad_y;

	return imagesBorders;
}







