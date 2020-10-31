
#include <pyramid/ChannelsPyramid.h>
#include <pyramid/ChannelsPyramidComputeAllStrategy.h>
#include <pyramid/ChannelsPyramidComputeAllParallelStrategy.h>
#include <pyramid/ChannelsPyramidApproximatedStrategy.h>
#include <pyramid/ChannelsPyramidApproximatedParallelStrategy.h>

#include <opencv2/opencv.hpp>
#include <iostream>
#include "gtest/gtest.h"

class TestChannelsPyramid: public testing::Test
{
public:
  std::shared_ptr<ChannelsPyramid> pChnsPyramid;
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

  pChnsPyramid = ChannelsPyramid::createChannelsPyramid("all", "pdollar");
  pChnsPyramid->getScales(nPerOct, nOctUp, minDS, shrink, size, scales, scaleshw);
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

  pChnsPyramid = ChannelsPyramid::createChannelsPyramid("all", "pdollar");

  pChnsPyramid->getScales(nPerOct, nOctUp, minDS, shrink, size, scales, scaleshw);
  pChnsPyramid->getScales(nPerOct, nOctUp, minDS, shrink, size, scales, scaleshw);
  std::vector<float> check = {1.0667, 0.9333, 0.8000, 0.6667, 0.5333};

  for(uint i = 0; i < scales.size(); i++){
    ASSERT_TRUE(abs(scales[i]-check[i])<1.e-3f);
  }
}

TEST_F(TestChannelsPyramid, channelsPyramidComputeAllStrategy)
{
  cv::Mat image = cv::imread("images/index.jpeg", cv::IMREAD_COLOR);

  std::string nameOpts = "yaml/pPyramid_badacost.yml";

  ClassifierConfig clfData;

  clfData.padding.width = 6; //
  clfData.padding.height = 4; //
  clfData.nOctUp = 0; //
  clfData.nPerOct = 3; //
  clfData.nApprox = 2; //
  clfData.shrink = 4;

  clfData.luv.smooth = false;
  clfData.luv.smooth_kernel_size = 1;

  clfData.gradMag.normRad = 5; 
  clfData.gradMag.normConst = 0.005; 

  clfData.gradHist.binSize = 2;
  clfData.gradHist.nOrients = 6;
  clfData.gradHist.softBin = 1;
  clfData.gradHist.full = false;

  float lmbds[3] = { 0.000000, 0.073930, 0.072470};
  for(int i = 0; i < 3; i++)
    clfData.lambdas.push_back(lmbds[i]);
  clfData.minDs.width = 48; 
  clfData.minDs.height = 84; 

  pChnsPyramid = ChannelsPyramid::createChannelsPyramid("all", "pdollar");


  //bool loadOk = pChnsPyramid->load(nameOpts.c_str());
  //ASSERT_TRUE(loadOk);
  std::vector<cv::Mat> filters; // empty filters is ACF pyramid.
  std::vector<double> scales;
  std::vector<cv::Size2d> scaleshw;
  std::vector<std::vector<cv::Mat>> pyramid = pChnsPyramid->compute(image, filters, scales, scaleshw, clfData);
  //ASSERT_TRUE(pyramid.size()==28);
}


TEST_F(TestChannelsPyramid, channelsPyramidComputeAllParallelStrategy)
{
  cv::Mat image = cv::imread("images/index.jpeg", cv::IMREAD_COLOR);

  std::string nameOpts = "yaml/pPyramid_badacost.yml";

  pChnsPyramid = ChannelsPyramid::createChannelsPyramid("all_parallel", "pdollar");

  ClassifierConfig clfData;

  clfData.padding.width = 6; //
  clfData.padding.height = 4; //
  clfData.nOctUp = 0; //
  clfData.nPerOct = 3; //
  clfData.nApprox = 2; //
  clfData.shrink = 4;

  clfData.luv.smooth = false;
  clfData.luv.smooth_kernel_size = 1;

  clfData.gradMag.normRad = 5; 
  clfData.gradMag.normConst = 0.005; 

  clfData.gradHist.binSize = 2;
  clfData.gradHist.nOrients = 6;
  clfData.gradHist.softBin = 1;
  clfData.gradHist.full = false;

  float lmbds[3] = { 0.000000, 0.073930, 0.072470};
  for(int i = 0; i < 3; i++)
    clfData.lambdas.push_back(lmbds[i]);
  clfData.minDs.width = 48; 
  clfData.minDs.height = 84; 

  //bool loadOk = pChnsPyramid->load(nameOpts.c_str());
  //ASSERT_TRUE(loadOk);
  std::vector<cv::Mat> filters; // empty filters is ACF pyramid.
  std::vector<double> scales;
  std::vector<cv::Size2d> scaleshw;
  std::vector<std::vector<cv::Mat>> pyramid = pChnsPyramid->compute(image, filters, scales, scaleshw, clfData);
  //ASSERT_TRUE(pyramid.size()==28);
}

TEST_F(TestChannelsPyramid, channelsPyramidApproximatedParallelStrategy)
{
  cv::Mat image = cv::imread("images/index.jpeg", cv::IMREAD_COLOR);

  std::string nameOpts = "yaml/pPyramid_badacost.yml";

  pChnsPyramid = ChannelsPyramid::createChannelsPyramid("approximated_parallel", "pdollar");

  ClassifierConfig clfData;

  clfData.padding.width = 6; //
  clfData.padding.height = 4; //
  clfData.nOctUp = 0; //
  clfData.nPerOct = 3; //
  clfData.nApprox = 2; //
  clfData.shrink = 4;

  clfData.luv.smooth = false;
  clfData.luv.smooth_kernel_size = 1;

  clfData.gradMag.normRad = 5; 
  clfData.gradMag.normConst = 0.005; 

  clfData.gradHist.binSize = 2;
  clfData.gradHist.nOrients = 6;
  clfData.gradHist.softBin = 1;
  clfData.gradHist.full = false;

  float lmbds[3] = { 0.000000, 0.073930, 0.072470};
  for(int i = 0; i < 3; i++)
    clfData.lambdas.push_back(lmbds[i]);
  clfData.minDs.width = 48; 
  clfData.minDs.height = 84; 

  //bool loadOk = pChnsPyramid->load(nameOpts.c_str());
  //ASSERT_TRUE(loadOk);
  std::vector<cv::Mat> filters; // empty filters is ACF pyramid.
  std::vector<double> scales;
  std::vector<cv::Size2d> scaleshw;
  auto start = std::chrono::system_clock::now(); 
  std::vector<std::vector<cv::Mat>> pyramid = pChnsPyramid->compute(image, filters, scales, scaleshw, clfData);
  //ASSERT_TRUE(pyramid.size()==28);
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<float,std::milli> duration = end - start;
  std::cout << duration.count() << "ms" << std::endl;
}

TEST_F(TestChannelsPyramid, channelsPyramidApproximatedStrategy)
{
  cv::Mat image = cv::imread("images/index.jpeg", cv::IMREAD_COLOR);

  std::string nameOpts = "yaml/pPyramid_badacost.yml";

  pChnsPyramid = ChannelsPyramid::createChannelsPyramid("approximated", "pdollar");

  ClassifierConfig clfData;

  clfData.padding.width = 6; //
  clfData.padding.height = 4; //
  clfData.nOctUp = 0; //
  clfData.nPerOct = 3; //
  clfData.nApprox = 2; //
  clfData.shrink = 4;

  clfData.luv.smooth = false;
  clfData.luv.smooth_kernel_size = 1;

  clfData.gradMag.normRad = 5; 
  clfData.gradMag.normConst = 0.005; 

  clfData.gradHist.binSize = 2;
  clfData.gradHist.nOrients = 6;
  clfData.gradHist.softBin = 1;
  clfData.gradHist.full = false;

  float lmbds[3] = { 0.000000, 0.073930, 0.072470};
  for(int i = 0; i < 3; i++)
    clfData.lambdas.push_back(lmbds[i]);
  clfData.minDs.width = 48; 
  clfData.minDs.height = 84; 

  //bool loadOk = pChnsPyramid->load(nameOpts.c_str());
  //ASSERT_TRUE(loadOk);
  std::vector<cv::Mat> filters; // empty filters is ACF pyramid.
  std::vector<double> scales;
  std::vector<cv::Size2d> scaleshw;
  auto start = std::chrono::system_clock::now(); 
  std::vector<std::vector<cv::Mat>> pyramid = pChnsPyramid->compute(image, filters, scales, scaleshw,clfData);
  //ASSERT_TRUE(pyramid.size()==28);
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<float,std::milli> duration = end - start;
  std::cout << duration.count() << "ms" << std::endl;
}

