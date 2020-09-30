/** ------------------------------------------------------------------------
 *
 *  @brief Test of Image Resample
 *  @author Jorge Vela Peña
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2020/06/02
 *
 *  ------------------------------------------------------------------------ */


#include <detectors/BadacostDetector.h>
#include <pyramid/ChannelsPyramidApproximatedStrategy.h>
#include <pyramid/ChannelsPyramidComputeAllParallelStrategy.h>
#include "gtest/gtest.h"
#include <opencv2/opencv.hpp>

#include <iostream>


class TestBadacostDetector: public testing::Test
{
public:
  virtual void SetUp()
    {
    }

  virtual void TearDown()
    {
    }
};

TEST_F(TestBadacostDetector, TestDetectorPyramidComputeAllStrategy)
{
  std::string clfPath = "yaml/clf.yml";
  std::string pyrPath = "yaml/pPyramid.yml";
  std::string filtersPath = "yaml/filterTest.yml";

  BadacostDetector badacost;
  bool loadVal = badacost.load(clfPath, pyrPath, filtersPath);
  ASSERT_TRUE(loadVal);

//    cv::Mat image = cv::imread("images/carretera.jpg", cv::IMREAD_COLOR);
  cv::Mat image = cv::imread("images/coches10.jpg", cv::IMREAD_COLOR);
//  cv::Mat image = cv::imread("images/coche_solo1.png", cv::IMREAD_COLOR);

  std::vector<DetectionRectangle> detections = badacost.detect(image);
  std::cout << detections;
  badacost.showResults(image, detections);
  cv::imshow("image", image);
  cv::waitKey();
}

TEST_F(TestBadacostDetector, TestDetectoryramidComputeAllParrallelStrategy){

  std::string clfPath = "yaml/clf.yml";
  std::string pyrPath = "yaml/pPyramid.yml";
  std::string filtersPath = "yaml/filterTest.yml";

  ChannelsPyramid* pPyramidStrategy = dynamic_cast<ChannelsPyramid*>( new ChannelsPyramidComputeAllParrallelStrategy() );
  BadacostDetector badacost(pPyramidStrategy);

  bool loadVal = badacost.load(clfPath, pyrPath, filtersPath);
  ASSERT_TRUE(loadVal);

//    cv::Mat image = cv::imread("images/carretera.jpg", cv::IMREAD_COLOR);
  cv::Mat image = cv::imread("images/coches10.jpg", cv::IMREAD_COLOR);
//  cv::Mat image = cv::imread("images/coche_solo1.png", cv::IMREAD_COLOR);

  std::vector<DetectionRectangle> detections = badacost.detect(image);
  std::cout << detections;
  badacost.showResults(image, detections);
//  cv::imshow("image", image);
//  cv::waitKey();

}


TEST_F(TestBadacostDetector, TestDetectorPyramidApproximatedStrategy){

  std::string clfPath = "yaml/clf.yml";
  std::string pyrPath = "yaml/pPyramid.yml";
  std::string filtersPath = "yaml/filterTest.yml";

  ChannelsPyramid* pPyramidStrategy = dynamic_cast<ChannelsPyramid*>( new ChannelsPyramidApproximatedStrategy() );
  BadacostDetector badacost(pPyramidStrategy);

  bool loadVal = badacost.load(clfPath, pyrPath, filtersPath);
  ASSERT_TRUE(loadVal);

//    cv::Mat image = cv::imread("images/carretera.jpg", cv::IMREAD_COLOR);
  cv::Mat image = cv::imread("images/coches10.jpg", cv::IMREAD_COLOR);
//  cv::Mat image = cv::imread("images/coche_solo1.png", cv::IMREAD_COLOR);

  std::vector<DetectionRectangle> detections = badacost.detect(image);
  std::cout << detections;
  badacost.showResults(image, detections);
//  cv::imshow("image", image);
//  cv::waitKey();

}













