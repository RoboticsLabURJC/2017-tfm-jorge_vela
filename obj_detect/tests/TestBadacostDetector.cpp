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

#undef DEBUG
//#define DEBUG

class TestBadacostDetector: public testing::Test
{
public:
  std::vector<DetectionRectangle> gt_detections;

  virtual void SetUp()
    {
      DetectionRectangle d1;
      d1.bbox.x = 105;
      d1.bbox.y = 246;
      d1.bbox.width = 125;
      d1.bbox.height = 76;
      d1.score = 22.5936;
      d1.class_index = 16;
      gt_detections.push_back(d1);

      DetectionRectangle d2;
      d2.bbox.x = 370;
      d2.bbox.y = 230;
      d2.bbox.width = 70;
      d2.bbox.height = 60;
      d2.score = 18.8303;
      d2.class_index = 7;
      gt_detections.push_back(d2);

      DetectionRectangle d3;
      d3.bbox.x = 544;
      d3.bbox.y = 240;
      d3.bbox.width = 96;
      d3.bbox.height = 60;
      d3.score = 15.8058;
      d3.class_index = 8;
      gt_detections.push_back(d3);

      DetectionRectangle d4;
      d4.bbox.x = 476;
      d4.bbox.y = 225;
      d4.bbox.width = 70;
      d4.bbox.height = 60;
      d4.score = 15.3124;
      d4.class_index = 7;
      gt_detections.push_back(d4);
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

  cv::Mat image = cv::imread("images/coches10.jpg", cv::IMREAD_COLOR);

  std::vector<DetectionRectangle> detections = badacost.detect(image);

#ifdef DEBUG
  std::cout << detections;
  badacost.showResults(image, detections);
  cv::imshow("image", image);
  cv::waitKey();
#endif

  // We declare the test is passed if all the gt detections have sufficient overlap
  // with one detected rectangle.
  for (const DetectionRectangle& gt_d: gt_detections)
  {
      bool oneOverlaps = false;
      for (const DetectionRectangle& d: detections)
      {
        float overlap = gt_d.overlap(d);
        oneOverlaps = ( overlap > 0.9);
        if (oneOverlaps)
        {
          break;
        }
      }
      ASSERT_TRUE(oneOverlaps);
  }
}

TEST_F(TestBadacostDetector, TestDetectoryramidComputeAllParallelStrategy){

  std::string clfPath = "yaml/clf.yml";
  std::string pyrPath = "yaml/pPyramid.yml";
  std::string filtersPath = "yaml/filterTest.yml";

  ChannelsPyramid* pPyramidStrategy = dynamic_cast<ChannelsPyramid*>( new ChannelsPyramidComputeAllParallelStrategy() );
  BadacostDetector badacost(pPyramidStrategy);

  bool loadVal = badacost.load(clfPath, pyrPath, filtersPath);
  ASSERT_TRUE(loadVal);

  cv::Mat image = cv::imread("images/coches10.jpg", cv::IMREAD_COLOR);
  std::vector<DetectionRectangle> detections = badacost.detect(image);

#ifdef DEBUG
  std::cout << detections;
  badacost.showResults(image, detections);
  cv::imshow("image", image);
  cv::waitKey();
#endif

  // We declare the test is passed if all the gt detections have sufficient overlap
  // with one detected rectangle.
  for (const DetectionRectangle& gt_d: gt_detections)
  {
      bool oneOverlaps = false;
      for (const DetectionRectangle& d: detections)
      {
        float overlap = gt_d.overlap(d);
        oneOverlaps = ( overlap > 0.9);
        if (oneOverlaps)
        {
          break;
        }
      }
      ASSERT_TRUE(oneOverlaps);
  }

}


TEST_F(TestBadacostDetector, TestDetectorPyramidApproximatedStrategy){

  std::string clfPath = "yaml/clf.yml";
  std::string pyrPath = "yaml/pPyramid.yml";
  std::string filtersPath = "yaml/filterTest.yml";

  ChannelsPyramid* pPyramidStrategy = dynamic_cast<ChannelsPyramid*>( new ChannelsPyramidApproximatedStrategy() );
  BadacostDetector badacost(pPyramidStrategy);

  bool loadVal = badacost.load(clfPath, pyrPath, filtersPath);
  ASSERT_TRUE(loadVal);

  cv::Mat image = cv::imread("images/coches10.jpg", cv::IMREAD_COLOR);

  std::vector<DetectionRectangle> detections = badacost.detect(image);

#ifdef DEBUG
  std::cout << detections;
  badacost.showResults(image, detections);
  cv::imshow("image", image);
  cv::waitKey();
#endif

  // We declare the test is passed if all the gt detections have sufficient overlap
  // with one detected rectangle.
  for (const DetectionRectangle& gt_d: gt_detections)
  {
      bool oneOverlaps = false;
      for (const DetectionRectangle& d: detections)
      {
        float overlap = gt_d.overlap(d);
        oneOverlaps = ( overlap > 0.9);
        if (oneOverlaps)
        {
          break;
        }
      }
      ASSERT_TRUE(oneOverlaps);
  }
}













