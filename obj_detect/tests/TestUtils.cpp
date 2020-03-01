/** ------------------------------------------------------------------------
 *
 *  @brief Test of Image Resample
 *  @author Jorge Vela Peña
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2019/07/08
 *
 *  ------------------------------------------------------------------------ */


#include <channels/Utils.h>

#include "gtest/gtest.h"
#include <opencv/cv.hpp>

#include <iostream>


using namespace cv;
using namespace std;


class TestUtils: public testing::Test
{
 public:
  virtual void SetUp()
  {
  }

  virtual void TearDown()
  {
  }
};


TEST_F(TestUtils, TestResample)
{
  cv::Mat image = cv::imread("index.jpeg", cv::IMREAD_COLOR); 
  int h = 40;
  int w = 37;
  cv:Mat dst = ImgResample(image, w, h, 3);
  //printf("%d %d \n", dst.size().width, dst.size().height);
}


TEST_F(TestUtils, TestChannelsCompute)
{
  cv::Mat image = cv::imread("index.jpeg", cv::IMREAD_COLOR); 
  int h = 40;
  int w = 37;
  channelsCompute(image, 3);
}


