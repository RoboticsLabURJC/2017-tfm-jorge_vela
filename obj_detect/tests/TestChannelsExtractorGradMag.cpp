/** ------------------------------------------------------------------------
 *
 *  @brief Test of Channels Extractor for magnitude and orient gradients.
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2019/07/08
 *
 *  ------------------------------------------------------------------------ */

#include <channels/ChannelsExtractorGradMagPDollar.h>
#include <channels/ChannelsExtractorGradMagOpenCV.h>
#include <channels/Utils.h>
#include "gtest/gtest.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <string>
#include <chrono> 

#undef DEBUG
//#define DEBUG

class TestChannelsExtractorGradMag: public testing::Test
{
public:
  virtual void SetUp()
    {
    }

  virtual void TearDown()
    {
    }

  void
  compareGradientMagnitudeAndOrientation
    (
    cv::Mat img,
    std::string matlab_grad_yaml_filename,
    std::string impl_type = "pdollar"
    );
};

void
TestChannelsExtractorGradMag::compareGradientMagnitudeAndOrientation
  (
  cv::Mat img,
  std::string matlab_grad_yaml_filename,
  std::string impl_type
  )
{
  cv::FileStorage fs;
  bool file_exists = fs.open(matlab_grad_yaml_filename, cv::FileStorage::READ);
  ASSERT_TRUE(file_exists);

  // Read matlab gradient magnitude parameters from yaml file
  float normConst = readScalarFromFileNode(fs["normConst"]);
  float normRad = readScalarFromFileNode(fs["normRad"]);

  ChannelsExtractorGradMag* pExtractor;
  if (impl_type == "opencv")
  {
    pExtractor = dynamic_cast<ChannelsExtractorGradMag*>(new ChannelsExtractorGradMagOpenCV(normRad, normConst));
  }
  else // "pdollar"
  {
    pExtractor = dynamic_cast<ChannelsExtractorGradMag*>(new ChannelsExtractorGradMagPDollar(normRad, normConst));
  }

  // Extract the gradient magnitude and orientation channels
  std::vector<cv::Mat> gradMagExtractVector;
  gradMagExtractVector = pExtractor->extractFeatures(img);
  delete pExtractor;

  // Get the matlab magnitude matrix from disk
  cv::Mat MatlabMag = readMatrixFromFileNode(fs["M"]);

  // Compare Matlab gradient magnitude with C++ implementation
  double min_val;
  double max_val;
  int min_ind[2];
  int max_ind[2];
  cv::minMaxIdx(MatlabMag, &min_val, &max_val, min_ind, max_ind, cv::Mat());
  float channel_range = max_val - min_val;
  float threshold = 0.05*channel_range; // Asume a 5% of the maximum value is an acceptable error.
  cv::Mat absDiff = cv::abs(gradMagExtractVector[0] - MatlabMag);
  cv::Mat lessThanThr = (absDiff < threshold)/255.0; // Boolean matrix has 255 for true and 0 for false.

#ifdef DEBUG
  cv::imshow("cpp-Mag", gradMagExtractVector[0]);
  cv::imshow("matlab-Mag", MatlabMag);
  cv::waitKey();
#endif

  int num_pixels_ok = cv::sum(lessThanThr)[0];
//  std::cout << "num_pixels_ok = " << num_pixels_ok << std::endl;
  ASSERT_TRUE(num_pixels_ok > 0.9 * absDiff.size().height * absDiff.size().height);

  // Get the matlab magnitude matrix from disk
  cv::Mat MatlabO = readMatrixFromFileNode(fs["O"]);

  // Compare Matlab gradient orientation with C++ implementation
  cv::minMaxIdx(MatlabMag, &min_val, &max_val, min_ind, max_ind, cv::Mat());
  channel_range = max_val - min_val;
  threshold = 0.05*channel_range; // Asume a 5% of the maximum value is an acceptable error.
  absDiff = cv::abs(gradMagExtractVector[1] - MatlabO);
  lessThanThr = (absDiff < threshold)/255.0; // Boolean matrix has 255 for true and 0 for false.

#ifdef DEBUG
  cv::imshow("cpp-O", gradMagExtractVector[1]);
  cv::imshow("matlab-O", MatlabO);
  cv::waitKey();
#endif

  num_pixels_ok = cv::sum(lessThanThr)[0];
  ASSERT_TRUE(num_pixels_ok > 0.9 * absDiff.size().height * absDiff.size().height);

}

TEST_F(TestChannelsExtractorGradMag, TestCompleteImageColor1PDollar)
{
  cv::Mat image;
  image = cv::imread("images/index.jpeg", cv::IMREAD_COLOR);
  ASSERT_TRUE(image.data);
  compareGradientMagnitudeAndOrientation(image,
                                         "yaml/index_jpeg_GradientChannels.yaml");
}


TEST_F(TestChannelsExtractorGradMag, TestCompleteImageGrayPDollar)
{
  cv::Mat image;
  image = cv::imread("images/index.jpeg", cv::IMREAD_GRAYSCALE);
  ASSERT_TRUE(image.data);
  compareGradientMagnitudeAndOrientation(image,
                                         "yaml/index_jpeg_gray_GradientChannels.yaml");
}

TEST_F(TestChannelsExtractorGradMag, TestCompleteImageColorNormConst0_07PDollar)
{
  cv::Mat image;
  image = cv::imread("images/index.jpeg", cv::IMREAD_COLOR);
  ASSERT_TRUE(image.data);
  compareGradientMagnitudeAndOrientation(image,
                                         "yaml/index_jpeg_GradientChannels_normConst_0_07.yaml");
}

TEST_F(TestChannelsExtractorGradMag, TestCompleteImageColorNormRad0PDollar)
{
  cv::Mat image;
  image = cv::imread("images/index.jpeg", cv::IMREAD_COLOR);
  ASSERT_TRUE(image.data);
  compareGradientMagnitudeAndOrientation(image,
                                         "yaml/index_jpeg_GradientChannels_normRad_0.yaml");
}

TEST_F(TestChannelsExtractorGradMag, TestCompleteImageColor1OpenCV)
{
  cv::Mat image;
  image = cv::imread("images/index.jpeg", cv::IMREAD_COLOR);
  ASSERT_TRUE(image.data);
  compareGradientMagnitudeAndOrientation(image,
                                         "yaml/index_jpeg_GradientChannels.yaml",
                                         "opencv");
}


TEST_F(TestChannelsExtractorGradMag, TestCompleteImageGrayOpenCV)
{
  cv::Mat image;
  image = cv::imread("images/index.jpeg", cv::IMREAD_GRAYSCALE);
  ASSERT_TRUE(image.data);
  compareGradientMagnitudeAndOrientation(image,
                                         "yaml/index_jpeg_gray_GradientChannels.yaml",
                                         "opencv");
}

TEST_F(TestChannelsExtractorGradMag, TestCompleteImageColorNormConst0_07OpenCV)
{
  cv::Mat image;
  image = cv::imread("images/index.jpeg", cv::IMREAD_COLOR);
  ASSERT_TRUE(image.data);
  compareGradientMagnitudeAndOrientation(image,
                                         "yaml/index_jpeg_GradientChannels_normConst_0_07.yaml",
                                         "opencv");
}

TEST_F(TestChannelsExtractorGradMag, TestCompleteImageColorNormRad0OpenCV)
{
  cv::Mat image;
  image = cv::imread("images/index.jpeg", cv::IMREAD_COLOR);
  ASSERT_TRUE(image.data);
  compareGradientMagnitudeAndOrientation(image,
                                         "yaml/index_jpeg_GradientChannels_normRad_0.yaml",
                                         "opencv");
}
