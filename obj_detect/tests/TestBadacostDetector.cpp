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
#include <pyramid/ChannelsPyramidApproximatedParallelStrategy.h>

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

#include <iostream>

#undef DEBUG
//#define DEBUG

#define OVERLAP_THRESHOLD 0.9

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

  virtual void
  testDetector
    (
    cv::Mat& image,
    BadacostDetector& detector
    );
};

TEST_F(TestBadacostDetector, TestNms)
{
  DetectionRectangle d1;
  d1.bbox.x = 568;
  d1.bbox.y = 432;
  d1.bbox.width = 123;
  d1.bbox.height = 233;
  d1.score = 1;
  d1.class_index = 1;
  gt_detections.push_back(d1);

  DetectionRectangle d2;
  d2.bbox.x = 570;
  d2.bbox.y = 422;
  d2.bbox.width = 120;
  d2.bbox.height = 233;
  d2.score = 1;
  d2.class_index = 1;
  gt_detections.push_back(d2);

  DetectionRectangle d3;
  d3.bbox.x = 573;
  d3.bbox.y = 435;
  d3.bbox.width = 118;
  d3.bbox.height = 220;
  d3.score = 1;
  d3.class_index = 1;
  gt_detections.push_back(d3);

  std::vector<DetectionRectangle> dv;
  dv.push_back(d1);
  dv.push_back(d2);
  dv.push_back(d3);

  std::vector<DetectionRectangle> dvo;
  BadacostDetector badacost;
  nonMaximumSuppression(dv, dvo);

  ASSERT_TRUE(dvo[0].bbox.x == 568);
  ASSERT_TRUE(dvo[0].bbox.y == 432);
  ASSERT_TRUE(dvo[0].bbox.width == 123);
  ASSERT_TRUE(dvo[0].bbox.height == 233);
}


TEST_F(TestBadacostDetector, TestNms2)
{
  DetectionRectangle d1;
  d1.bbox.x = 334;
  d1.bbox.y = 334;
  d1.bbox.width = 222;
  d1.bbox.height = 233;
  d1.score = 1;
  d1.class_index = 1;
  gt_detections.push_back(d1);

  DetectionRectangle d2;
  d2.bbox.x = 142;
  d2.bbox.y = 543;
  d2.bbox.width = 333;
  d2.bbox.height = 20;
  d2.score = 1;
  d2.class_index = 1;
  gt_detections.push_back(d2);

  DetectionRectangle d3;
  d3.bbox.x = 330;
  d3.bbox.y = 210;
  d3.bbox.width = 222;
  d3.bbox.height = 433;
  d3.score = 1;
  d3.class_index = 1;
  gt_detections.push_back(d3);

  std::vector<DetectionRectangle> dv;
  dv.push_back(d1);
  dv.push_back(d2);
  dv.push_back(d3);

  std::vector<DetectionRectangle> dvo;
  BadacostDetector badacost;
  nonMaximumSuppression(dv, dvo);

  ASSERT_TRUE(dvo[0].bbox.x == 334);
  ASSERT_TRUE(dvo[0].bbox.y == 334);
  ASSERT_TRUE(dvo[0].bbox.width == 222);
  ASSERT_TRUE(dvo[0].bbox.height == 233);

  ASSERT_TRUE(dvo[1].bbox.x == 142);
  ASSERT_TRUE(dvo[1].bbox.y == 543);
  ASSERT_TRUE(dvo[1].bbox.width == 333);
  ASSERT_TRUE(dvo[1].bbox.height == 20);
}

void
TestBadacostDetector::testDetector
  (
    cv::Mat& image,
    BadacostDetector& detector
  )
{
  std::vector<DetectionRectangle> detections = detector.detect(image);

#ifdef DEBUG
  std::cout << "detections = " << std::endl;
  std::cout << detections;
  std::cout << "gt_detections = " << std::endl;
  std::cout << gt_detections << std::endl;
  detector.showResults(image, detections);
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
        oneOverlaps = ( overlap >= OVERLAP_THRESHOLD);
        if (oneOverlaps)
        {
          break;
        }
      }
      ASSERT_TRUE(oneOverlaps);
  }
}

// --------------------------------------------------------------------------
//  P.Dollar ACF implementation
// --------------------------------------------------------------------------

TEST_F(TestBadacostDetector, TestDetectorPyramidComputeAllStrategyPDollar)
{
  std::string clfPath = "yaml/detectorComplete_2.yml";
  std::string filtersPath = "yaml/filterTest.yml";

  BadacostDetector badacost("all", "pdollar");
  bool loadVal = badacost.load(clfPath, filtersPath);
  ASSERT_TRUE(loadVal);

  cv::Mat image = cv::imread("images/coches10.jpg", cv::IMREAD_COLOR);

  testDetector(image, badacost);
}

TEST_F(TestBadacostDetector, TestDetectoryramidComputeAllParallelStrategyPDollar)
{
  std::string clfPath = "yaml/detectorComplete_2.yml";
  std::string filtersPath = "yaml/filterTest.yml";

  BadacostDetector badacost("all_parallel", "pdollar");
  bool loadVal = badacost.load(clfPath, filtersPath); //, pyrPath (segundo parametro)
  ASSERT_TRUE(loadVal);
  cv::Mat image = cv::imread("images/coches10.jpg", cv::IMREAD_COLOR);

  testDetector(image, badacost);
}


TEST_F(TestBadacostDetector, TestDetectorPyramidApproximatedStrategyPDollar)
{
  std::string clfPath = "yaml/detectorComplete_2.yml";
  std::string filtersPath = "yaml/filterTest.yml";

  BadacostDetector badacost("approximated", "pdollar");
  bool loadVal = badacost.load(clfPath, filtersPath); //, pyrPath (segundo parametro)
  ASSERT_TRUE(loadVal);
  cv::Mat image = cv::imread("images/coches10.jpg", cv::IMREAD_COLOR);

  testDetector(image, badacost);
}


TEST_F(TestBadacostDetector, TestDetectorPyramidApproximatedParallelStrategyPDollar)
{
  std::string clfPath = "yaml/detectorComplete_2.yml";
  std::string filtersPath = "yaml/filterTest.yml";

  BadacostDetector badacost("approximated_parallel", "pdollar");
  bool loadVal = badacost.load(clfPath, filtersPath); //, pyrPath (segundo parametro)
  ASSERT_TRUE(loadVal);
  cv::Mat image = cv::imread("images/coches10.jpg", cv::IMREAD_COLOR);

  testDetector(image, badacost);
}


// --------------------------------------------------------------------------
//  OpenCV ACF implementation
// --------------------------------------------------------------------------

TEST_F(TestBadacostDetector, TestDetectorPyramidComputeAllStrategyOpenCV)
{
  std::string clfPath = "yaml/detectorComplete_2.yml";
  std::string filtersPath = "yaml/filterTest.yml";

  BadacostDetector badacost("all", "opencv");
  bool loadVal = badacost.load(clfPath, filtersPath);
  ASSERT_TRUE(loadVal);

  cv::Mat image = cv::imread("images/coches10.jpg", cv::IMREAD_COLOR);

  testDetector(image, badacost);
}

TEST_F(TestBadacostDetector, TestDetectoryramidComputeAllParallelStrategyOpenCV)
{
  std::string clfPath = "yaml/detectorComplete_2.yml";
  std::string filtersPath = "yaml/filterTest.yml";

  BadacostDetector badacost("all_parallel", "opencv");
  bool loadVal = badacost.load(clfPath, filtersPath); //, pyrPath (segundo parametro)
  ASSERT_TRUE(loadVal);
  cv::Mat image = cv::imread("images/coches10.jpg", cv::IMREAD_COLOR);

  testDetector(image, badacost);
}


TEST_F(TestBadacostDetector, TestDetectorPyramidApproximatedStrategyOpenCV)
{
  std::string clfPath = "yaml/detectorComplete_2.yml";
  std::string filtersPath = "yaml/filterTest.yml";

  BadacostDetector badacost("approximated", "opencv");
  bool loadVal = badacost.load(clfPath, filtersPath); //, pyrPath (segundo parametro)
  ASSERT_TRUE(loadVal);
  cv::Mat image = cv::imread("images/coches10.jpg", cv::IMREAD_COLOR);

  testDetector(image, badacost);
}


TEST_F(TestBadacostDetector, TestDetectorPyramidApproximatedParallelStrategyOpenCV)
{
  std::string clfPath = "yaml/detectorComplete_2.yml";
  std::string filtersPath = "yaml/filterTest.yml";

  BadacostDetector badacost("approximated_parallel", "opencv");
  bool loadVal = badacost.load(clfPath, filtersPath); //, pyrPath (segundo parametro)
  ASSERT_TRUE(loadVal);
  cv::Mat image = cv::imread("images/coches10.jpg", cv::IMREAD_COLOR);

  testDetector(image, badacost);
}










