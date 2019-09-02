#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

#include "ChannelsLUVExtractor.h"
#include "ChannelsBordersExtractor.h"

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
    if( argc != 2)
    {
     cout <<" Usage: display_image ImageToLoadAndDisplay" << endl;
     return -1;
    }

    cv::Mat image;
    image = imread(argv[1], CV_LOAD_IMAGE_COLOR);   // Read the file

    if(! image.data )                              // Check for invalid input
    {
        cout << "Could not open or find the image" << std::endl ;
        return -1;
    }

	ChannelsLUVExtractor chluv;
	std::vector<cv::Mat> LUV = chluv.extractFeatures(image);
	/*imshow( "Image L " , LUV[0] );
	imshow( "Image U " , LUV[1] );
	imshow( "Image V " , LUV[2] );

    waitKey(0);*/
    Mat imageGRAY;
    cvtColor(image, imageGRAY, CV_BGR2GRAY); 
    cout << "Width : " << image.cols << endl;
	cout << "Height: " << image.rows << endl;
	
	Mat imageBlackGRAY;
	Mat imageblack(image.rows, image.cols, CV_8UC3, Scalar(0,0,0));
    cvtColor(imageblack, imageBlackGRAY, CV_BGR2GRAY); 

	/*for(int i = 0; i <273; i++){
			
	}*/
	
     /*//Para detectar bordes
    //1. Convertir a escala de grises
	Mat imageGRAY;
	Mat borde;
	cvtColor(image, imageGRAY, CV_BGR2GRAY); 
	cv::Canny(imageGRAY,borde,0,30,3);
	
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;

	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;
	/// Gradient X
	Sobel( imageGRAY, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
	/// Gradient Y
	Sobel( imageGRAY, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
		
    namedWindow( "Display window X", WINDOW_AUTOSIZE );
    convertScaleAbs( grad_x, abs_grad_x );
    imshow( "Display window X", abs_grad_x ); 
    
    namedWindow( "Display window Y", WINDOW_AUTOSIZE );
    convertScaleAbs( grad_y, abs_grad_y );
    imshow( "Display window Y", abs_grad_y ); 
    
    
    //Gradiente total
    Mat grad;
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
    imshow( "Total gradient" , grad );
    waitKey(0);*/

	// Para obtener imágenes a 45 grados realizar una convolución total.
    return 0;
}














