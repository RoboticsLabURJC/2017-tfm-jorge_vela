/** ------------------------------------------------------------------------
 *
 *  @brief Test of Channels Extractor for LUV color space
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2019/07/08
 *
 *  ------------------------------------------------------------------------ */

#include <channels/ChannelsExtractorGradMag.h>
#include "gtest/gtest.h"
#include <opencv/cv.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <string>


using namespace cv;
using namespace std;

#undef VISUALIZE_RESULTS

class TestChannelsExtractorGradMag: public testing::Test
{
public:
  const int ROWS = 2;
  const int COLS = 4;
  GradMagExtractor gradMagExtract;

  virtual void SetUp()
    {
    }

  virtual void TearDown()
    {
    }
};


TEST_F(TestChannelsExtractorGradMag, TestGradMagUpRight)
{
  const int h=12, w=12  , misalign=1; int x, y, d; //192   const int h=12, w=12,
  d = 3;
  float *O, *M;
  int size = h*w*d;
  int sizeData = sizeof(float);

  float I[h*w*3], *I0=I+misalign;
  for( x=0; x<h*w*3; x++ ) I0[x]=0;
  I0[0] = 1;
  
  shared_ptr<float> punt_mat(new float);
  M = new float[size](); 
  O = new float[size]();

  gradMagExtract.gradM(I0, M, O);

  ASSERT_TRUE(abs(M[0] - 1.4146) < 1.e-4f);
  ASSERT_TRUE(abs(M[1] - 0.5001) < 1.e-4f);
  ASSERT_TRUE(abs(M[12] - 0.5001) < 1.e-4f);

  ASSERT_TRUE(abs(O[0] - 0.7857) < 1.e-4f);
  ASSERT_TRUE(abs(O[1] - 1.5708) < 1.e-4f);
  ASSERT_TRUE(abs(O[12] - 3.1171) < 1.e-4f);

  M = NULL; O = NULL;
  free(M); free(O);
}


TEST_F(TestChannelsExtractorGradMag, TestGradMagDownLeft)
{
  const int h=12, w=12  , misalign=1; int x, y, d; //192   const int h=12, w=12,
  d = 3;
  float *M, *O, *H;
  int size = h*w*d;
  int sizeData = sizeof(float);

  float I[h*w*3], *I0=I+misalign;
  for( x=0; x<h*w*3; x++ ) I0[x]=0;
  I0[143] = 1;
  
  M = new float[size](); // (size, sizeData, misalign)??
  O = new float[size]();

  gradMagExtract.gradM(I0, M, O);

  ASSERT_TRUE(abs(M[143] - 1.4146) < 1.e-4f);
  ASSERT_TRUE(abs(M[142] - 0.5001) < 1.e-4f);
  ASSERT_TRUE(abs(M[131] - 0.5001) < 1.e-4f);

  ASSERT_TRUE(abs(O[143] - 0.7857) < 1.e-4f);
  ASSERT_TRUE(abs(O[142] - 1.5708) < 1.e-4f);
  ASSERT_TRUE(abs(O[131] - 0.0245) < 1.e-4f);

  M = NULL; O = NULL;
  free(M); free(O);
}


TEST_F(TestChannelsExtractorGradMag, TestGradMagCenter)
{
  const int h=12, w=12  , misalign=1; int x, y, d; //192   const int h=12, w=12,
  d = 3;
  float *M, *O, *H;
  int size = h*w*d;
  int sizeData = sizeof(float);

  float I[h*w*3], *I0=I+misalign;
  for( x=0; x<h*w*3; x++ ) I0[x]=0;
  I0[76] = 1;
  
  M = new float[size](); // (size, sizeData, misalign)??
  O = new float[size]();

  gradMagExtract.gradM(I0, M, O);

  ASSERT_TRUE(abs(M[77] - 0.5001) < 1.e-4f);
  ASSERT_TRUE(abs(M[75] - 0.5001) < 1.e-4f);
  ASSERT_TRUE(abs(M[64] - 0.5001) < 1.e-4f);
  ASSERT_TRUE(abs(M[88] - 0.5001) < 1.e-4f);

  ASSERT_TRUE(abs(O[77] - 1.5708) < 1.e-4f);
  ASSERT_TRUE(abs(O[75] - 1.5708) < 1.e-4f);
  ASSERT_TRUE(abs(O[64] - 0.0245) < 1.e-4f);
  ASSERT_TRUE(abs(O[88] - 3.1171) < 1.e-4f);

  M = NULL; O = NULL;
  free(M); free(O);
}

TEST_F(TestChannelsExtractorGradMag, TestGradMagLinear)
{
  const int h=12, w=12  , misalign=1; int x, y, d; //192   const int h=12, w=12,
  d = 3;
  float *M, *O, *H;
  int size = h*w*d;
  int sizeData = sizeof(float);


  float I[h*w*3], *I0=I+misalign;
  for( x=0; x<h*w*3; x++ ) I0[x]=0;
  for(x = 60; x < 72; x++) I0[x] = 1;
 
  M = new float[size](); // (size, sizeData, misalign)??
  O = new float[size]();

  gradMagExtract.gradM(I0, M, O);


  for(y=48; y < 59;  y++)
  {
    ASSERT_TRUE(abs(M[y] - 0.5001) < 1.e-4f);
  }

  for(y=72; y < 83;  y++)
  {
    ASSERT_TRUE(abs(M[y] - 0.5001) < 1.e-4f);
  }

  for(y=48; y < 59;  y++)
  {
    ASSERT_TRUE(abs(O[y] - 0.0245) < 1.e-4f);
  }

  for(y=72; y < 83;  y++)
  {
    ASSERT_TRUE(abs(O[y] - 3.1171) < 1.e-4f);
  }

  M = NULL; O = NULL;
  free(M); free(O);
}


/*
TEST_F(TestChannelsExtractorGradMag, TestCompleteImage)
{
  cv::Mat image;
  image = cv::imread("images/index2.jpeg", cv::IMREAD_GRAYSCALE); //IMREAD_COLOR);

  float *M, *O, *H;

  int size = image.cols*image.rows*1;
  int sizeData = sizeof(float);

  M = new float[size](); // (size, sizeData, misalign)??
  O = new float[size]();
	
  gradMagExtract.gradMAdv(image,M,O);
  FileStorage fs;
  fs.open("yaml/MRealImage.yaml", FileStorage::READ);

  FileNode rows = fs["M"]["rows"];
  FileNode cols = fs["M"]["cols"];
  FileNode MMatrix = fs["M"]["data"];

  int i = 0;
  for(int y=0;y<(int)rows;y++)
  { 
    for(int x=0;x<(int)cols;x++)
    {
      ASSERT_TRUE(abs(M[x*(int)cols+y] - (float)MMatrix[i]) < 1.e-4f);
      i++;	
    } 
  }

  fs.release();
  fs.open("yaml/ORealImage.yaml", FileStorage::READ);

  rows = fs["O"]["rows"];
  cols = fs["O"]["cols"];
  FileNode OMatrix = fs["O"]["data"];	

  i = 0;
  for(int y=0;y<(int)rows;y++)
  { 
    for(int x=0;x<(int)cols;x++)
    {
      ASSERT_TRUE(abs(O[x*(int)cols+y] - (float)OMatrix[i]) < 1.e-4f);
      i++;	
    } 
  }

  M = NULL; O = NULL;
  free(M); free(O);	
}

TEST_F(TestChannelsExtractorGradMag, TestCompleteImageColor)
{
  cv::Mat image;
  image = cv::imread("images/index3.jpeg", cv::IMREAD_COLOR); //IMREAD_COLOR);

  float *M, *O, *H;

  int size = image.cols*image.rows*3;
  int sizeData = sizeof(float);

  M = new float[size](); 
  O = new float[size]();
	
  gradMagExtract.gradMAdv(image,M,O);

  FileStorage fs;
  fs.open("yaml/M_colorScale.yaml", FileStorage::READ);

  FileNode rows = fs["M"]["rows"];
  FileNode cols = fs["M"]["cols"];
  FileNode MMatrix = fs["M"]["data"];

  int tot = (int)rows*(int)cols;
  for(int i=0; i < tot; i++)
  {
    float valorMatlab = (float)MMatrix[i];
    ASSERT_TRUE(abs(M[i] - valorMatlab) < 1.e-4f);
  }

  free(M); free(O);	
}
*/


