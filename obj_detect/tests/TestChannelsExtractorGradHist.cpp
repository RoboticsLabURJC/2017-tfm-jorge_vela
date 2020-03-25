/** ------------------------------------------------------------------------
 *
 *  @brief Test of Channels Extractor for LUV color space
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2019/07/08
 *
 *  ------------------------------------------------------------------------ */

#include <channels/ChannelsExtractorGradHist.h>
#include <channels/ChannelsExtractorGradMag.h>

#include "gtest/gtest.h"
#include <opencv/cv.hpp>

#include <iostream>


using namespace cv;
using namespace std;

#undef VISUALIZE_RESULTS

class TestChannelsExtractorGradHist: public testing::Test
{
 public:

  const int ROWS = 2;
  const int COLS = 4;
  GradHistExtractor gradHistExtract;

  GradMagExtractor gradMagExtract;
  virtual void SetUp()
  {
  }

  virtual void TearDown()
  {
  }
};

/*
TEST_F(TestChannelsExtractorGradHist, TestMexImage)
{
	const int h=12, w=12  , misalign=1; int x, y, d; //192   const int h=12, w=12,
	d = 3;
	float *M, *O, *H;
	int size = h*w*d;
	int sizeData = sizeof(float);


  	float I[h*w*3], *I0=I+misalign;
  	for( x=0; x<h*w*3; x++ ) I0[x]=0;
  	I0[0] = 1;


	M = gradHistExtract.allocW(size, sizeData, misalign);
	O = gradHistExtract.allocW(size, sizeData, misalign);

	//Para calcular H, h2=h/4 w2=w/4, sizeTotal*6
	int h2 = h/4;
	int w2 = w/4;
	int sizeH = h2*w2*d*6; //Valor obtenido de su ejemplo chnsTestCpp.cpp
	H = gradHistExtract.allocW(sizeH, sizeData, misalign);

	gradMagExtract.gradM(I0, M, O);
	//float M1 =  M[0];
	//float ExpectedValue = 1.4146;
	//EXPECT_EQ(M[0], M[0]);
 	gradHistExtract.gradH(M, O, H);
	
 	//channExtract.HOG(M, O, H);

 	//printf("---------------- M: ----------------\n");
	//for(y=0;y<h;y++){ for(x=0;x<w;x++) printf("%.4f ",M[x*h+y]); printf("\n");}
	//printf("---------------- O: ----------------\n");
	//for(y=0;y<h;y++){ for(x=0;x<w;x++) printf("%.4f ",O[x*h+y]); printf("\n");}
	//printf("---------------- O: ----------------\n");
	//for(y=0;y<h2;y++){ for(x=0;x<w2;x++) printf("%.4f ",H[x*h2+y]); printf("\n");}

	//printf("%f \n", H[0]);

	//EXPECT_TRUE(M1==ExpectedValue);* /
}


TEST_F(TestChannelsExtractorGradHist, TestRealImage){
	cv::Mat image;
	image = cv::imread("index3.jpeg", cv::IMREAD_GRAYSCALE); //IMREAD_COLOR);
	float *M, *O, *H;

    int misalign=1;
	int h = image.size().height;
	int w = image.size().width;
    int nChannels = image.channels();

	int size = h*w*nChannels;
	int sizeData = sizeof(float);


	//M = gradHistExtract.allocW(size, sizeData, misalign);
	//O = gradHistExtract.allocW(size, sizeData, misalign);
	M = new float[size](); // (size, sizeData, misalign)??
	O = new float[size]();

	int h2 = h/4;
	int w2 = w/4;
	int sizeH = h2*w2*nChannels*6; //Valor obtenido de su ejemplo chnsTestCpp.cpp
	H = new float[sizeH]();


	FileStorage fs;
    fs.open("MRealImage.yaml", FileStorage::READ);
    FileNode MMatrix = fs["M"]["data"];
    int i = 0;

    FileNode rows = fs["M"]["rows"];
    FileNode cols = fs["M"]["cols"];


	for(int y=0;y<(int)rows;y++){ 
		for(int x=0;x<(int)cols;x++){
			M[x*(int)cols+y] = (float)MMatrix[i];
			i++;	
		} 
	}

    fs.open("ORealImage.yaml", FileStorage::READ);

    rows = fs["O"]["rows"];
    cols = fs["O"]["cols"];
    FileNode OMatrix = fs["O"]["data"];	

	i = 0;
	for(int y=0;y<(int)rows;y++){ 
		for(int x=0;x<(int)cols;x++){
			O[x*(int)cols+y] = (float)OMatrix[i];
			i++;	
		} 
	}

	//printf("%.4f\n", H[0]);

	gradHistExtract.gradHAdv(image, M, O, H);

	//printf("%.4f\n", H[0]);
}
*/


TEST_F(TestChannelsExtractorGradHist, TestColorImage){
	cv::Mat image;
	image = cv::imread("index3.jpeg", cv::IMREAD_COLOR); 

	int size = image.size().height*image.size().width*1;
	float *M = new float[size](); // (size, sizeData, misalign)??
	float *O = new float[size]();

	float *H= new float[size*6]();

	gradMagExtract.gradMAdv(image,M,O);

	printf("%.4f %.4f\n", M[0], O[0] );
	gradHistExtract.gradHAdv(image, M, O, H);
}



