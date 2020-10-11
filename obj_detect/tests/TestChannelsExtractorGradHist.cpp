/** ------------------------------------------------------------------------
 *
 *  @brief Test of Channels Extractor for histogram gradients.
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2019/07/08
 *
 *  ------------------------------------------------------------------------ */

#include <channels/ChannelsExtractorGradHistPDollar.h>
#include <channels/ChannelsExtractorGradMagPDollar.h>
#include <channels/Utils.h>

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
  ChannelsExtractorGradMag* pGradMagExtract;
  ChannelsExtractorGradHist* pGradHistExtract;
  ChannelsExtractorGradHist* pGradHistExtractBinSizeOrients;

  virtual void SetUp()
  {
    pGradMagExtract =
              dynamic_cast<ChannelsExtractorGradMag*>(new ChannelsExtractorGradMagPDollar());
    pGradHistExtract =
              dynamic_cast<ChannelsExtractorGradHist*>(new ChannelsExtractorGradHistPDollar());
    pGradHistExtractBinSizeOrients =
              dynamic_cast<ChannelsExtractorGradHist*>(new ChannelsExtractorGradHistPDollar(6,9,1,0));
  }

  virtual void TearDown()
  {
    delete pGradMagExtract;
    delete pGradHistExtract;
    delete pGradHistExtractBinSizeOrients;
  }
};

TEST_F(TestChannelsExtractorGradHist, TestColorImagePDollar){
  cv::Mat image;
  image = cv::imread("images/index3.jpeg", cv::IMREAD_COLOR); 

  std::vector<cv::Mat> gradMagExtractVector(2);
  gradMagExtractVector = pGradMagExtract->extractFeatures(image);


//  auto start = std::chrono::high_resolution_clock::now();
  std::vector<cv::Mat> gradHistExtractVector;
  gradHistExtractVector = pGradHistExtract->extractFeatures(image,gradMagExtractVector );
//  auto stop = std::chrono::high_resolution_clock::now();
//  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
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

//  cv::Mat H1_matlab = readMatrixFromFileNodeWrongBufferMatlab(fs["H"]);
//  transpose(H1_matlab, H1_matlab);
//  H1_matlab = H1_matlab(cv::Range(0, height), cv::Range(0, width));
////  transpose(H1_matlab, H1_matlab);

  cv::FileNode rows = fs["H"]["rows"];
  cv::FileNode cols = fs["H"]["cols"];
  cv::FileNode HMatrix = fs["H"]["data"];

  int i = 0;
  int num_pixels_ok = 0;
  for(int y=0; y < width; y++)
  {
    for(int x=0; x < height; x++)
    {
      if (abs(dirH1[x*width+y] - (float)HMatrix[i]) < 1.e-1f)
      {
        num_pixels_ok++;
      }
      i++;
    }
  }
//  std::cout << "H1 = " << H1 << std::endl;
//  std::cout << "H1_matlab = " << H1_matlab << std::endl;

//  cv::Mat absDiff;
//  absDiff = cv::abs(H1 - H1_matlab);
//  cv::Mat belowTh = (absDiff < 1.e-1f)/255;
//  int num_pixels_ok = cv::sum(belowTh)[0];
//  std::cout << "num_pixels_ok = " << num_pixels_ok << std::endl;
  ASSERT_TRUE(num_pixels_ok > 0.7 * height * width);
}


TEST_F(TestChannelsExtractorGradHist, TestColorImageBinSizeOrientsPDollar){
  cv::Mat image;
  image = cv::imread("images/index3.jpeg", cv::IMREAD_COLOR); 

  std::vector<cv::Mat> gradMagExtractVector(2);
  gradMagExtractVector = pGradMagExtract->extractFeatures(image);

  std::vector<cv::Mat> gradHistExtractVector;
  gradHistExtractVector = pGradHistExtractBinSizeOrients->extractFeatures(image,gradMagExtractVector);

  int height = gradHistExtractVector[0].size().height;
  int width = gradHistExtractVector[0].size().width;

  cv::FileStorage fs;  
  bool file_exists = fs.open("yaml/TestHColorBinSizeOrients.yml", cv::FileStorage::READ);
  ASSERT_TRUE(file_exists);

  cv::FileNode rows = fs["H"]["rows"];
  cv::FileNode cols = fs["H"]["cols"];
  cv::FileNode HMatrix = fs["H"]["data"];

  int i = 0;
  for(int j=0; j < 3; j++)
  {
    cv::Mat H1;
    gradHistExtractVector[j].convertTo(H1, CV_32F);  
    float *dirH1 = H1.ptr<float>();
    int num_pixels_ok = 0;
    for(int y=0;y < width; y++)
    { 
      for(int x=0; x < height; x++)
      {
        if (abs(dirH1[x*(int)width+y] - (float)HMatrix[i]) < 1.e-1f)
        {
          num_pixels_ok++;
        }
        i++;
      } 
    }
    ASSERT_TRUE(num_pixels_ok > 0.8* height * width);
  }


}










