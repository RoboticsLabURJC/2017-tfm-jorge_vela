/** ------------------------------------------------------------------------
 *
 *  @brief Test of Channels Extractor for magnitude and orient gradients.
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2019/07/08
 *
 *  ------------------------------------------------------------------------ */

#include <channels/ChannelsExtractorGradMag.h>
#include <channels/Utils.h>
#include "gtest/gtest.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <string>
#include <chrono> 

#undef SHOW_CHANNELS
//#define SHOW_CHANNELS

class TestChannelsExtractorGradMag: public testing::Test
{
public:
  const int ROWS = 2;
  const int COLS = 4;

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
    std::string matlab_grad_mag_yaml_filename,
    std::string matlab_grad_orient_yaml_filename
    );
};

void
TestChannelsExtractorGradMag::compareGradientMagnitudeAndOrientation
  (
  cv::Mat img,
  std::string matlab_grad_mag_yaml_filename,
  std::string matlab_grad_orient_yaml_filename
  )
{
  cv::FileStorage fs;
  bool file_exists = fs.open(matlab_grad_mag_yaml_filename, cv::FileStorage::READ);
  ASSERT_TRUE(file_exists);

  // Read matlab gradient magnitude parameters from yaml file
//  cv::FileNode data = fs["normRad"]["data"];
//  std::vector<float> p;
//  data >> p;
//  float normRad = p[0];

//  data = fs["normConst"]["data"];
//  p.clear();
//  data >> p;
//  float normConst = p[0];

  float normConst = readScalarFromFileNode(fs["normConst"]);
  float normRad = readScalarFromFileNode(fs["normRad"]);

  GradMagExtractor extractor(normRad, normConst);

  // Extract the gradient magnitude and orientation channels
  std::vector<cv::Mat> gradMagExtractVector;
  gradMagExtractVector = extractor.extractFeatures(img);

//  int rows = static_cast<int>(fs["M"]["rows"]);
//  int cols = static_cast<int>(fs["M"]["cols"]);
//  cv::Mat MatlabMag = cv::Mat::zeros(rows, cols, CV_32F);
//  data = fs["M"]["data"];
//  p.clear();
//  data >> p;
//  memcpy(MatlabMag.data, p.data(), p.size()*sizeof(float));
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

#ifdef SHOW_CHANNELS
  cv::imshow("cpp-Mag", gradMagExtractVector[0]);
  cv::imshow("matlab-Mag", MatlabMag);
  cv::waitKey();
#endif

  int num_pixels_ok = cv::sum(lessThanThr)[0];
//  std::cout << "num_pixels_ok = " << num_pixels_ok << std::endl;
  ASSERT_TRUE(num_pixels_ok > 0.9 * absDiff.size().height * absDiff.size().height);
  fs.release();

  // Read matlab gradient orientation from yaml file.
  file_exists = fs.open(matlab_grad_orient_yaml_filename, cv::FileStorage::READ);
  ASSERT_TRUE(file_exists);

//  rows = static_cast<int>(fs["O"]["rows"]);
//  cols = static_cast<int>(fs["O"]["cols"]);
//  cv::Mat MatlabO = cv::Mat::zeros(rows, cols, CV_32F);
//  data = fs["O"]["data"];
//  p.clear();
//  data >> p;
//  memcpy(MatlabO.data, p.data(), p.size()*sizeof(float));

  cv::Mat MatlabO = readMatrixFromFileNode(fs["O"]);

  // Compare Matlab gradient orientation with C++ implementation
  cv::minMaxIdx(MatlabMag, &min_val, &max_val, min_ind, max_ind, cv::Mat());
  channel_range = max_val - min_val;
  threshold = 0.05*channel_range; // Asume a 5% of the maximum value is an acceptable error.
  absDiff = cv::abs(gradMagExtractVector[1] - MatlabO);
  lessThanThr = (absDiff < threshold)/255.0; // Boolean matrix has 255 for true and 0 for false.

#ifdef SHOW_CHANNELS
  cv::imshow("cpp-O", gradMagExtractVector[1]);
  cv::imshow("matlab-O", MatlabO);
  cv::waitKey();
#endif

  num_pixels_ok = cv::sum(lessThanThr)[0];
//  std::cout << "num_pixels_ok = " << num_pixels_ok << std::endl;
  ASSERT_TRUE(num_pixels_ok > 0.9 * absDiff.size().height * absDiff.size().height);
}

TEST_F(TestChannelsExtractorGradMag, TestCompleteImageColor1)
{
  cv::Mat image;
  image = cv::imread("images/index.jpeg", cv::IMREAD_COLOR);
  ASSERT_TRUE(image.data);
  compareGradientMagnitudeAndOrientation(image,
                                         "yaml/index_jpeg_GradientChannels.yaml",
                                         "yaml/index_jpeg_GradientChannels.yaml");
}


TEST_F(TestChannelsExtractorGradMag, TestCompleteImageGray)
{
  cv::Mat image;
  image = cv::imread("images/index.jpeg", cv::IMREAD_GRAYSCALE);
  ASSERT_TRUE(image.data);
  compareGradientMagnitudeAndOrientation(image,
                                         "yaml/index_jpeg_gray_GradientChannels.yaml",
                                         "yaml/index_jpeg_gray_GradientChannels.yaml");
}

TEST_F(TestChannelsExtractorGradMag, TestCompleteImageColorNormConst0_07)
{
  cv::Mat image;
  image = cv::imread("images/index.jpeg", cv::IMREAD_COLOR);
  ASSERT_TRUE(image.data);
  compareGradientMagnitudeAndOrientation(image,
                                         "yaml/index_jpeg_gray_GradientChannels_normConst_0_07.yaml",
                                         "yaml/index_jpeg_gray_GradientChannels_normConst_0_07.yaml");
}

TEST_F(TestChannelsExtractorGradMag, TestCompleteImageColorNormRad0)
{
  cv::Mat image;
  image = cv::imread("images/index.jpeg", cv::IMREAD_COLOR);
  ASSERT_TRUE(image.data);
  compareGradientMagnitudeAndOrientation(image,
                                         "yaml/index_jpeg_gray_GradientChannels_normRad_0.yaml",
                                         "yaml/index_jpeg_gray_GradientChannels_normRad_0.yaml");
}
