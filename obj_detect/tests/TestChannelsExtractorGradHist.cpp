/** ------------------------------------------------------------------------
 *
 *  @brief Test of Channels Extractor for histogram gradients.
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2019/07/08
 *
 *  ------------------------------------------------------------------------ */

#include <channels/ChannelsExtractorGradHist.h>
#include <channels/ChannelsExtractorGradMag.h>

#include "gtest/gtest.h"
#include <opencv2/opencv.hpp>

#include <iostream>
#include <chrono>


using namespace cv;
using namespace std;

#undef VISUALIZE_RESULTS

class TestChannelsExtractorGradHist: public testing::Test
{
 public:

  const int ROWS = 2;
  const int COLS = 4;
  GradMagExtractor gradMagExtract;

  GradHistExtractor gradHistExtract;
  GradHistExtractor gradHistExtractBinSizeOrients{6,9,1,0};



  virtual void SetUp()
  {
  }

  virtual void TearDown()
  {
  }
};

TEST_F(TestChannelsExtractorGradHist, TestColorImage){
  cv::Mat image;
  image = cv::imread("images/index3.jpeg", cv::IMREAD_COLOR); 

  std::vector<cv::Mat> gradMagExtractVector(2);
  gradMagExtractVector = gradMagExtract.extractFeatures(image);


  auto start = std::chrono::high_resolution_clock::now();

  std::vector<cv::Mat> gradHistExtractVector;
  gradHistExtractVector = gradHistExtract.extractFeatures(image,gradMagExtractVector );

  auto stop = std::chrono::high_resolution_clock::now(); 
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start); 
  //std::cout << "time ms: " << duration.count() << std::endl;


  int height = gradHistExtractVector[0].size().height;
  int width = gradHistExtractVector[0].size().width;

  cv::Mat H1;
  gradHistExtractVector[0].convertTo(H1, CV_32F);  
  float *dirH1 = H1.ptr<float>();

  cv::Mat H2;
  gradHistExtractVector[1].convertTo(H2, CV_32F);  
  float *dirH2 = H2.ptr<float>();

  cv::FileStorage fs;
  bool file_exists = fs.open("yaml/TestHColorScale.yml", cv::FileStorage::READ);
  ASSERT_TRUE(file_exists);

  cv::FileNode rows = fs["H"]["rows"];
  cv::FileNode cols = fs["H"]["cols"];
  cv::FileNode HMatrix = fs["H"]["data"];

  int i = 0;
  for(int y=0;y<(int)width;y++)
  { 
    for(int x=0;x<(int)height;x++)
    {
      ASSERT_TRUE(abs(dirH1[x*(int)width+y] - (float)HMatrix[i]) < 1.e-1f);
      i++;  
    } 
  }

}


TEST_F(TestChannelsExtractorGradHist, TestColorImageBinSizeOrients){
  cv::Mat image;
  image = cv::imread("images/index3.jpeg", cv::IMREAD_COLOR); 

  std::vector<cv::Mat> gradMagExtractVector(2);
  gradMagExtractVector = gradMagExtract.extractFeatures(image);

  std::vector<cv::Mat> gradHistExtractVector;
  gradHistExtractVector = gradHistExtractBinSizeOrients.extractFeatures(image,gradMagExtractVector);

  int height = gradHistExtractVector[0].size().height;
  int width = gradHistExtractVector[0].size().width;

  cv::FileStorage fs;  
  bool file_exists = fs.open("yaml/TestHColorBinSizeOrients.yml", cv::FileStorage::READ);
  ASSERT_TRUE(file_exists);

  cv::FileNode rows = fs["H"]["rows"];
  cv::FileNode cols = fs["H"]["cols"];
  cv::FileNode HMatrix = fs["H"]["data"];

  int i = 0;
  for(int j=0; j < 3; j++){
    printf("%d\n", j);
    cv::Mat H1;
    gradHistExtractVector[j].convertTo(H1, CV_32F);  
    float *dirH1 = H1.ptr<float>();
    for(int y=0;y<(int)width;y++)
    { 
      for(int x=0;x<(int)height;x++)
      {
        //printf("%.4f %.4f\n", dirH1[x*(int)width+y], (float)HMatrix[i]);
        ASSERT_TRUE(abs(dirH1[x*(int)width+y] - (float)HMatrix[i]) < 1.e-1f);
        i++;  
      } 
    }
  }
}










