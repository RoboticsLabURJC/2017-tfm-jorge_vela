/** ------------------------------------------------------------------------
 *
 *  @brief Test of Channels Extractor for magnitude and orient gradients.
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2019/07/08
 *
 *  ------------------------------------------------------------------------ */

#include <channels/ChannelsExtractorGradMag.h>
#include "gtest/gtest.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <string>
#include <chrono> 

#undef VISUALIZE_RESULTS

class TestChannelsExtractorGradMag: public testing::Test
{
public:
  const int ROWS = 2;
  const int COLS = 4;
  GradMagExtractor gradMagExtract;
  GradMagExtractor gradMagExtractNorm{5};
  GradMagExtractor gradMagExtractNormConst{5, 0.07};

  virtual void SetUp()
    {
    }

  virtual void TearDown()
    {
    }
};


TEST_F(TestChannelsExtractorGradMag, TestCompleteImageGray)
{
  cv::Mat image;
  image = cv::imread("images/index2.jpeg", cv::IMREAD_GRAYSCALE); //IMREAD_COLOR);

  int size = image.cols*image.rows*1;
  int sizeData = sizeof(float);

  std::vector<cv::Mat> gradMagExtractVector;
  gradMagExtractVector = gradMagExtract.extractFeatures(image);
  cv::Mat newM, newO;
  gradMagExtractVector[0].convertTo(newM, CV_32F);    
  float *M = newM.ptr<float>();

  gradMagExtractVector[1].convertTo(newO, CV_32F);    
  float *O = newO.ptr<float>();

  cv::FileStorage fs;
  fs.open("yaml/TestMGrayScale.yml", cv::FileStorage::READ);

  cv::FileNode rows = fs["M"]["rows"];
  cv::FileNode cols = fs["M"]["cols"];
  cv::FileNode MMatrix = fs["M"]["data"];

  int i = 0;
  for(int y=0;y<(int)rows;y++)
  { 
    for(int x=0;x<(int)cols;x++)
    {
      ASSERT_TRUE(abs(M[x*(int)cols+y] - (float)MMatrix[i]) < 1.e-3f);
      i++;	
    } 
  }

  fs.release();
  fs.open("yaml/TestOGrayScale.yml", cv::FileStorage::READ);

  rows = fs["O"]["rows"];
  cols = fs["O"]["cols"];
  cv::FileNode OMatrix = fs["O"]["data"];	

  i = 0;
  for(int y=0;y<(int)rows;y++)
  { 
    for(int x=0;x<(int)cols;x++)
    {
      ASSERT_TRUE(abs(O[x*(int)cols+y] - (float)OMatrix[i]) < 1.e-3f);
      i++;	
    } 
  }

  M = NULL; O = NULL;
  free(M); free(O);	
}


TEST_F(TestChannelsExtractorGradMag, TestCompleteImageColor)
{
  cv::Mat image;
  image = cv::imread("images/index2.jpeg", cv::IMREAD_COLOR); //IMREAD_COLOR);

  int size = image.cols*image.rows*3;
  int sizeData = sizeof(float);

  std::vector<cv::Mat> gradMagExtractVector;
  gradMagExtractVector = gradMagExtract.extractFeatures(image);
  cv::Mat newM, newO;
  gradMagExtractVector[0].convertTo(newM, CV_32F);    
  float *M = newM.ptr<float>();

  gradMagExtractVector[1].convertTo(newO, CV_32F);    
  float *O = newO.ptr<float>();

  cv::FileStorage fs;
  fs.open("yaml/TestMColorScale.yml", cv::FileStorage::READ);

  cv::FileNode rows = fs["M"]["rows"];
  cv::FileNode cols = fs["M"]["cols"];
  cv::FileNode MMatrix = fs["M"]["data"];

  int tot = image.cols*image.rows;// (int)rows*(int)cols;
  int i = 0;
  for(int y=0;y<(int)rows;y++)
  { 
    for(int x=0;x<(int)cols;x++)
    {
      ASSERT_TRUE(abs(M[x*(int)cols+y] - (float)MMatrix[i]) < 1.e-3f);
      i++;  
    } 
  }

  fs.open("yaml/TestOColorScale.yml", cv::FileStorage::READ);

  rows = fs["O"]["rows"];
  cols = fs["O"]["cols"];
  cv::FileNode OMatrix = fs["O"]["data"];

  i = 0;
  for(int y=0;y<(int)rows;y++)
  { 
    for(int x=0;x<(int)cols;x++)
    {
      ASSERT_TRUE(abs(O[x*(int)cols+y] - (float)OMatrix[i]) < 1.e-3f);
      i++;  
    } 
  }
  //free(M); free(O);	
}


TEST_F(TestChannelsExtractorGradMag, TestCompleteColorMagNorm)
{
  cv::Mat image;
  image = cv::imread("images/index3.jpeg", cv::IMREAD_COLOR); //IMREAD_COLOR);

  int size = image.cols*image.rows*3;
  int sizeData = sizeof(float);

  std::vector<cv::Mat> gradMagExtractVector;
  gradMagExtractVector = gradMagExtractNorm.extractFeatures(image);

  cv::Mat newM, newO;
  gradMagExtractVector[0].convertTo(newM, CV_32F);    
  float *M = newM.ptr<float>();

  gradMagExtractVector[1].convertTo(newO, CV_32F);    
  float *O = newO.ptr<float>();

    cv::FileStorage fs;
  fs.open("yaml/TestMColorNormRad.yml", cv::FileStorage::READ);

  cv::FileNode rows = fs["M"]["rows"];
  cv::FileNode cols = fs["M"]["cols"];
  cv::FileNode MMatrix = fs["M"]["data"];

  int tot = image.cols*image.rows;// (int)rows*(int)cols;
  int i = 0;
  for(int y=0;y<(int)rows;y++)
  { 
    for(int x=0;x<(int)cols;x++)
    {
      ASSERT_TRUE(abs(M[x*(int)cols+y] - (float)MMatrix[i]) < 1.e-3f);
      i++;  
    } 
  }

  fs.open("yaml/TestOColorNormRad.yml", cv::FileStorage::READ);

  rows = fs["O"]["rows"];
  cols = fs["O"]["cols"];
  cv::FileNode OMatrix = fs["O"]["data"];

  i = 0;
  for(int y=0;y<(int)rows;y++)
  { 
    for(int x=0;x<(int)cols;x++)
    {
      ASSERT_TRUE(abs(O[x*(int)cols+y] - (float)OMatrix[i]) < 1.e-3f);
      i++;  
    } 
  }
}


TEST_F(TestChannelsExtractorGradMag, TestCompleteColorMagNormConst)
{
  cv::Mat image;
  image = cv::imread("images/index3.jpeg", cv::IMREAD_COLOR); //IMREAD_COLOR);

  int size = image.cols*image.rows*3;
  int sizeData = sizeof(float);

  std::vector<cv::Mat> gradMagExtractVector;

  auto start = std::chrono::high_resolution_clock::now();
  
  gradMagExtractVector = gradMagExtractNormConst.extractFeatures(image);

  auto stop = std::chrono::high_resolution_clock::now(); 
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start); 
  std::cout << "time ms: " << duration.count() << std::endl;

  cv::Mat newM, newO;
  gradMagExtractVector[0].convertTo(newM, CV_32F);    
  float *M = newM.ptr<float>();

  gradMagExtractVector[1].convertTo(newO, CV_32F);    
  float *O = newO.ptr<float>();


  //cv::imshow("M",gradMagExtractVector[0]);
  //cv::imshow("O",gradMagExtractVector[1]);
  //cv::waitKey(0);


  cv::FileStorage fs;
  fs.open("yaml/TestMColorNormRadConst.yml", cv::FileStorage::READ);

  cv::FileNode rows = fs["M"]["rows"];
  cv::FileNode cols = fs["M"]["cols"];
  cv::FileNode MMatrix = fs["M"]["data"];

  int tot = image.cols*image.rows;// (int)rows*(int)cols;
  int i = 0;
  for(int y=0;y<(int)rows;y++)
  { 
    for(int x=0;x<(int)cols;x++)
    {
      ASSERT_TRUE(abs(M[x*(int)cols+y] - (float)MMatrix[i]) < 1.e-3f);
      i++;  
    } 
  }

  fs.open("yaml/TestOColorNormRadConst.yml", cv::FileStorage::READ);

  rows = fs["O"]["rows"];
  cols = fs["O"]["cols"];
  cv::FileNode OMatrix = fs["O"]["data"];

  i = 0;
  for(int y=0;y<(int)rows;y++)
  { 
    for(int x=0;x<(int)cols;x++)
    {
      ASSERT_TRUE(abs(O[x*(int)cols+y] - (float)OMatrix[i]) < 1.e-3f);
      i++;  
    } 
  }
}
