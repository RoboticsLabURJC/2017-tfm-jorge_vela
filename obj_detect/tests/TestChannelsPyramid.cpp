
#include <channels/ChannelsPyramid.h> 
#include "gtest/gtest.h"
#include <opencv2/opencv.hpp>

#include <iostream>


class TestChannelsPyramid: public testing::Test
{
public:
  ChannelsPyramid chnsPyramid;
  virtual void SetUp()
    {
    }

  virtual void TearDown()
    {
    }



};

TEST_F(TestChannelsPyramid, TestGetScales)
{
  int nPerOct = 8;
  int nOctUp = 1;
  int shrink = 4;
  cv::Size size;
  size.height = 19;
  size.width = 22;
  cv::Size minDS;
  size.height = 16;
  size.width = 16;
  std::vector<double> scales;
  std::vector<cv::Size2d> scaleshw;
  chnsPyramid.getScales(nPerOct, nOctUp, minDS, shrink, size, scales, scaleshw);
  std::vector<float> check = {2.1463, 1.8537, 1.6589, 1.4632, 1.2684, 1.0737, 0.8779};

  for(uint i = 0; i < scales.size(); i++){
    ASSERT_TRUE(abs(scales[i]-check[i])<1.e-3f);
  }
}

TEST_F(TestChannelsPyramid, TestGetScalesChangeVals)
{
  int nPerOct = 7;
  int nOctUp = 0;
  int shrink = 4;
  cv::Size size;
  size.height = 30;
  size.width = 30;
  cv::Size minDS;
  size.height = 16;
  size.width = 16;
  std::vector<double> scales;
  std::vector<cv::Size2d> scaleshw;
  chnsPyramid.getScales(nPerOct, nOctUp, minDS, shrink, size, scales, scaleshw);


  chnsPyramid.getScales(nPerOct, nOctUp, minDS, shrink, size, scales, scaleshw);
  std::vector<float> check = {1.0667, 0.9333, 0.8000, 0.6667, 0.5333};

  for(uint i = 0; i < scales.size(); i++){
    ASSERT_TRUE(abs(scales[i]-check[i])<1.e-3f);
  }
}


TEST_F(TestChannelsPyramid, channelsPyramid){
  cv::Mat image = cv::imread("images/index.jpeg", cv::IMREAD_COLOR);

  std::string nameOpts = "yaml/pPyramid.yml";
  bool loadOk = chnsPyramid.load(nameOpts.c_str());
  ASSERT_TRUE(loadOk);
  std::vector<cv::Mat> filters; // empty filters is ACF pyramid.
  std::vector<std::vector<cv::Mat>> pyramid = chnsPyramid.compute(image, filters);
  //ASSERT_TRUE(pyramid.size()==28);
}



TEST_F(TestChannelsPyramid, badacostFilters){
//  cv::Mat image = cv::imread("images/carretera.jpg", cv::IMREAD_COLOR);
  cv::Mat image = cv::imread("images/coches3.jpg", cv::IMREAD_COLOR);
  std::string nameOpts = "yaml/pPyramid.yml";
  bool loadOk = chnsPyramid.load(nameOpts.c_str());
  if (!loadOk)
  {
      throw std::runtime_error("TestChannelsPyramid: pPyramid.yml not found");
  }
  std::vector<cv::Mat> filters; // empty filters to compute ACF pyramid
  std::vector<std::vector<cv::Mat>> pyramid = chnsPyramid.compute(image, filters);

  /*std::vector<cv::Mat> filtered = chnsPyramid.badacostFilters(pyramid[11], "yaml/filterTest.yml");


  printf("%d %d \n", filtered[0].size().height, filtered[0].size().width);

  for(int k = 0; k < 40; k++){
    double minVal; 
    double maxVal; 
    cv::Point minLoc; 
    cv::Point maxLoc;
    minMaxLoc( filtered[k], &minVal, &maxVal, &minLoc, &maxLoc );
    std::cout << "min val: " << minVal << " max val:" << maxVal << std::endl;
  }*/
  //cv::imshow("", filtered[0]);
  //cv::waitKey(0);



}





 
