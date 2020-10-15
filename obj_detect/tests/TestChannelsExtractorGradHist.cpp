/** ------------------------------------------------------------------------
 *
 *  @brief Test of Channels Extractor for histogram gradients.
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2019/07/08
 *
 *  ------------------------------------------------------------------------ */

#include <channels/ChannelsExtractorGradHistPDollar.h>
#include <channels/ChannelsExtractorGradMagPDollar.h>
#include <channels/ChannelsExtractorGradHistOpenCV.h>
#include <channels/ChannelsExtractorGradMagOpenCV.h>
#include <channels/Utils.h>

#include "gtest/gtest.h"
#include <opencv2/opencv.hpp>

#include <iostream>
#include <chrono>


using namespace cv;
using namespace std;

#undef DEBUG
//#define DEBUG

class TestChannelsExtractorGradHist: public testing::Test
{
 public:

  virtual void SetUp()
  {
  }

  virtual void TearDown()
  {
  }

  void compareGradientOrientationHistogram
    (
    cv::Mat img,
    std::string matlab_grad_yaml_filename,
    std::string impl_type = "pdollar"
    );
};

void
TestChannelsExtractorGradHist::compareGradientOrientationHistogram
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
  float nOrients = readScalarFromFileNode(fs["nOrients"]);
  float binSize = readScalarFromFileNode(fs["binSize"]);
  float softBin = readScalarFromFileNode(fs["softBin"]);
  float full = readScalarFromFileNode(fs["full"]);

  ChannelsExtractorGradMag* pExtractorMag;
  ChannelsExtractorGradHist* pExtractorHist;
  if (impl_type == "opencv")
  {
    pExtractorMag = dynamic_cast<ChannelsExtractorGradMag*>(new ChannelsExtractorGradMagOpenCV(normRad, normConst));
    pExtractorHist = dynamic_cast<ChannelsExtractorGradHist*>(new ChannelsExtractorGradHistOpenCV(binSize, nOrients, softBin, full));
  }  
  else // "pdollar"
  {
    pExtractorMag = dynamic_cast<ChannelsExtractorGradMag*>(new ChannelsExtractorGradMagPDollar(normRad, normConst));
    pExtractorHist = dynamic_cast<ChannelsExtractorGradHist*>(new ChannelsExtractorGradHistPDollar(binSize, nOrients, softBin, full));
  }

  // Extract the gradient histogram channels
  std::vector<cv::Mat> gradMagExtractVector;
  std::vector<cv::Mat> gradHistExtractVector;
  gradMagExtractVector = pExtractorMag->extractFeatures(img);
  gradHistExtractVector = pExtractorHist->extractFeatures(img, gradMagExtractVector);
  delete pExtractorMag;
  delete pExtractorHist;

#ifdef DEBUG
  std::cout << "softBin = " << softBin << std::endl;
  std::cout << "binSize = " << binSize << std::endl;
#endif

  // Compare Matlab gradient histograms with C++ implementation
  for (uint i=0; i < gradHistExtractVector.size(); i++)
  {
    // Get the matlab magnitude matrix from disk
    std::string var_name = "H_" + std::to_string(i+1);
    cv::Mat MatlabMat = readMatrixFromFileNode(fs[var_name]);

#ifdef DEBUG
    std::cout << "var_name = " << var_name << std::endl;
//    std::cout << "MatlabMat.size() = " << MatlabMat.size() << std::endl;
//    std::cout << "gradHistExtractVector[i].size() = " << gradHistExtractVector[i].size() << std::endl;
#endif

    double min_val;
    double max_val;
    int min_ind[2];
    int max_ind[2];
    cv::minMaxIdx(MatlabMat, &min_val, &max_val, min_ind, max_ind, cv::Mat());
    float channel_range = max_val - min_val;
    float threshold = 0.05*channel_range; // Asume a 5% of the channel's range is an acceptable error.
    cv::Mat absDiff = cv::abs(gradHistExtractVector[i] - MatlabMat);
    cv::Mat lessThanThr = (absDiff < threshold)/255.0; // Boolean matrix has 255 for true and 0 for false.

#ifdef DEBUG
    cv::imshow("cpp-Mat", gradHistExtractVector[i]);
    cv::imshow("matlab-Mat", MatlabMat);
    cv::waitKey();
#endif

    int num_pixels_ok = cv::sum(lessThanThr)[0];
    ASSERT_TRUE(num_pixels_ok > 0.9 * absDiff.size().height * absDiff.size().height);
  }
  fs.release();
}

TEST_F(TestChannelsExtractorGradHist, TestCompleteImageColor1PDollar)
{
  cv::Mat image;
  image = cv::imread("images/index.jpeg", cv::IMREAD_COLOR);
  ASSERT_TRUE(image.data);
  compareGradientOrientationHistogram(image,
                                      "yaml/index_jpeg_GradientChannels.yaml");
}

TEST_F(TestChannelsExtractorGradHist, TestCompleteImageColorSoftBinNeg4PDollar)
{
  cv::Mat image;
  image = cv::imread("images/index.jpeg", cv::IMREAD_COLOR);
  ASSERT_TRUE(image.data);
  compareGradientOrientationHistogram(image,
                                      "yaml/index_jpeg_GradientChannels_softBin_negative_4.yaml",
                                      "pdollar");
}

TEST_F(TestChannelsExtractorGradHist, TestCompleteImageColorBinSize5PDollar)
{
  cv::Mat image;
  image = cv::imread("images/index.jpeg", cv::IMREAD_COLOR);
  ASSERT_TRUE(image.data);
  compareGradientOrientationHistogram(image,
                                      "yaml/index_jpeg_GradientChannels_binSize_5.yaml",
                                      "pdollar");
}

TEST_F(TestChannelsExtractorGradHist, TestCompleteImageColorBinSize1SoftBin2PDollar)
{
  cv::Mat image;
  image = cv::imread("images/index.jpeg", cv::IMREAD_COLOR);
  ASSERT_TRUE(image.data);
  compareGradientOrientationHistogram(image,
                                      "yaml/index_jpeg_GradientChannels_binSize_1_softBin_2.yaml",
                                      "pdollar");
}

TEST_F(TestChannelsExtractorGradHist, TestCompleteImageGrayPDollar)
{
  cv::Mat image;
  image = cv::imread("images/index.jpeg", cv::IMREAD_GRAYSCALE);
  ASSERT_TRUE(image.data);
  compareGradientOrientationHistogram(image,
                                      "yaml/index_jpeg_gray_GradientChannels.yaml");
}

TEST_F(TestChannelsExtractorGradHist, TestCompleteImageColorNormConst0_07PDollar)
{
  cv::Mat image;
  image = cv::imread("images/index.jpeg", cv::IMREAD_COLOR);
  ASSERT_TRUE(image.data);
  compareGradientOrientationHistogram(image,
                                      "yaml/index_jpeg_GradientChannels_normConst_0_07.yaml");
}

TEST_F(TestChannelsExtractorGradHist, TestCompleteImageColorNormRad0PDollar)
{
  cv::Mat image;
  image = cv::imread("images/index.jpeg", cv::IMREAD_COLOR);
  ASSERT_TRUE(image.data);
  compareGradientOrientationHistogram(image,
                                      "yaml/index_jpeg_GradientChannels_normRad_0.yaml");
}

TEST_F(TestChannelsExtractorGradHist, TestCompleteImageColor1OpenCV)
{
  cv::Mat image;
  image = cv::imread("images/index.jpeg", cv::IMREAD_COLOR);
  ASSERT_TRUE(image.data);
  compareGradientOrientationHistogram(image,
                                      "yaml/index_jpeg_GradientChannels.yaml",
                                      "opencv");
}


TEST_F(TestChannelsExtractorGradHist, TestCompleteImageColorSoftBinNeg4OpenCV)
{
  cv::Mat image;
  image = cv::imread("images/index.jpeg", cv::IMREAD_COLOR);
  ASSERT_TRUE(image.data);
  compareGradientOrientationHistogram(image,
                                      "yaml/index_jpeg_GradientChannels_softBin_negative_4.yaml",
                                      "opencv");
}

TEST_F(TestChannelsExtractorGradHist, TestCompleteImageColorBinSize5OpenCV)
{
  cv::Mat image;
  image = cv::imread("images/index.jpeg", cv::IMREAD_COLOR);
  ASSERT_TRUE(image.data);
  compareGradientOrientationHistogram(image,
                                      "yaml/index_jpeg_GradientChannels_binSize_5.yaml",
                                      "opencv");
}

TEST_F(TestChannelsExtractorGradHist, TestCompleteImageColorBinSize1SoftBin2OpenCV)
{
  cv::Mat image;
  image = cv::imread("images/index.jpeg", cv::IMREAD_COLOR);
  ASSERT_TRUE(image.data);
  compareGradientOrientationHistogram(image,
                                      "yaml/index_jpeg_GradientChannels_binSize_1_softBin_2.yaml",
                                      "opencv");
}

TEST_F(TestChannelsExtractorGradHist, TestCompleteImageGrayOpenCV)
{
  cv::Mat image;
  image = cv::imread("images/index.jpeg", cv::IMREAD_GRAYSCALE);
  ASSERT_TRUE(image.data);
  compareGradientOrientationHistogram(image,
                                      "yaml/index_jpeg_gray_GradientChannels.yaml",
                                      "opencv");
}

TEST_F(TestChannelsExtractorGradHist, TestCompleteImageColorNormConst0_07OpenCV)
{
  cv::Mat image;
  image = cv::imread("images/index.jpeg", cv::IMREAD_COLOR);
  ASSERT_TRUE(image.data);
  compareGradientOrientationHistogram(image,
                                      "yaml/index_jpeg_GradientChannels_normConst_0_07.yaml",
                                      "opencv");
}

TEST_F(TestChannelsExtractorGradHist, TestCompleteImageColorNormRad0OpenCV)
{
  cv::Mat image;
  image = cv::imread("images/index.jpeg", cv::IMREAD_COLOR);
  ASSERT_TRUE(image.data);
  compareGradientOrientationHistogram(image,
                                         "yaml/index_jpeg_GradientChannels_normRad_0.yaml",
                                         "opencv");
}














