/** ------------------------------------------------------------------------
 *
 *  @brief Test of Image Resample
 *  @author Jorge Vela Peña
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2020/06/02
 *
 *  ------------------------------------------------------------------------ */


#include <detectors/badacostDetector.h>
#include <pyramid/ChannelsPyramidApproximatedStrategy.h>
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

  std::vector<cv::Rect2i> detections = badacost.detect(image);
  for(uint i = 0; i < detections.size(); i++)
  {
    std::cout << "[ x=" << detections[i].x << ", y=";
    std::cout << detections[i].y << ", w=" << detections[i].width << ", h=" << detections[i].height;
    std::cout << " ] " << std::endl;
    cv::rectangle(image, detections[i], cv::Scalar(200,0,0),2);
  }
  std::cout << "detections.size() = " << detections.size() << std::endl;

  cv::imshow("image", image);
  cv::waitKey();
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

  std::vector<cv::Rect2i> detections = badacost.detect(image);
  for(uint i = 0; i < detections.size(); i++)
  {
    std::cout << "[ x=" << detections[i].x << ", y=";
    std::cout << detections[i].y << ", w=" << detections[i].width << ", h=" << detections[i].height;
    std::cout << " ] " << std::endl;
    cv::rectangle(image, detections[i], cv::Scalar(200,0,0),2);
  }
  std::cout << "detections.size() = " << detections.size() << std::endl;

  cv::imshow("image", image);
  cv::waitKey();

}













