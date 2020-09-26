/** ------------------------------------------------------------------------
 *
 *  @brief Test of Extraction of ACF Channels.
 *  @author Jorge Vela Peña
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2020/09/25
 *
 *  ------------------------------------------------------------------------ */
#include <channels/ChannelsExtractorACF.h>

#include "gtest/gtest.h"
#include <opencv2/opencv.hpp>

#include <iostream>


using namespace cv;
using namespace std;


class TestChannelsExtractorACF: public testing::Test
{
public:
  virtual void SetUp()
    {
    }

  virtual void TearDown()
    {
    }
};

TEST_F(TestChannelsExtractorACF, TestExtractChannelsACFColorImage)
{
  cv::Mat image = cv::imread("images/index3.jpeg", cv::IMREAD_COLOR);
  std::vector<cv::Mat> pChnsCompute;
  cv::Size padding;
  padding.width = 6;
  padding.height = 4;
  ChannelsExtractorACF acfExtractor(padding, 4, "RGB");
  pChnsCompute = acfExtractor.extractFeatures(image);

  cv::Mat testMag;
  transpose(pChnsCompute[3], testMag);

  cv::Mat imgMag;
  testMag.convertTo(imgMag, CV_32F);
  float *valuesImgMag = imgMag.ptr<float>();

  FileStorage fs1;
  fs1.open("yaml/TestMagChnsCompute.yml", FileStorage::READ);
  FileNode imgMagMatlab = fs1["M"]["data"];

  for(int i=0;i<14*17 /*(int)rows*(int)cols*/;i++)
  {
    //printf("%.5f %.5f \n", (float)valuesImgMag[i], (float)imgMagMatlab[i] );
    ASSERT_TRUE(abs((float)valuesImgMag[i] - (float)imgMagMatlab[i]) < 1.e-2f);
  }
}
