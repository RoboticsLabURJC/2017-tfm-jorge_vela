/** ------------------------------------------------------------------------
 *
 *  @brief Test of Image Resample
 *  @author Jorge Vela Peña
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2019/07/08
 *
 *  ------------------------------------------------------------------------ */


#include <channels/ImageResample.h>

#include "gtest/gtest.h"
#include <opencv/cv.hpp>

#include <iostream>


using namespace cv;
using namespace std;


class TestImageResample: public testing::Test
{
 public:
  ImageResample imgResample;
  virtual void SetUp()
  {
  }

  virtual void TearDown()
  {
  }
};

TEST_F(TestImageResample, TestImageResample)
{
	printf("HEEELLO WORLD\n");
	cv::Mat image = cv::imread("index.jpeg", cv::IMREAD_COLOR); 
	int h = 40;
	int w = 37;
	cv:Mat dst = imgResample.ImgResample(image, w, h, 3);
	printf("%d %d \n", dst.size().width, dst.size().height);

}



