/** ------------------------------------------------------------------------
 *
 *  @brief Test of Channels Extractor for magnitude and orient gradients.
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2019/07/08
 *
 *  ------------------------------------------------------------------------ */

#include <channels/ChannelsExtractorGradMagPDollar.h>
#include <channels/ChannelsExtractorGradMagOpenCV.h>
#include <channels/ChannelsExtractorGradMagOpenCL.h>
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
  else if (impl_type == "opencl")
  {
    pExtractor = dynamic_cast<ChannelsExtractorGradMag*>(new ChannelsExtractorGradMagOpenCL(normRad, normConst));
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
  int num_pixels_ok = cv::sum(lessThanThr)[0];

#ifdef DEBUG
  std::cout << "num_pixels_ok = " << num_pixels_ok;
  std::cout << " of " << absDiff.rows * absDiff.cols << std::endl;
  cv::imshow("absDiff", absDiff);
  cv::imshow("cpp-Mag", gradMagExtractVector[0]);
  cv::imshow("matlab-Mag", MatlabMag);
  cv::waitKey();
#endif

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
  num_pixels_ok = cv::sum(lessThanThr)[0];

#ifdef DEBUG
  std::cout << "num_pixels_ok = " << num_pixels_ok;
  std::cout << " of " << absDiff.rows * absDiff.cols << std::endl;
  cv::imshow("absDiff", absDiff);
  cv::imshow("cpp-O", gradMagExtractVector[1]);
  cv::imshow("matlab-O", MatlabO);
  cv::waitKey();
#endif

  ASSERT_TRUE(num_pixels_ok > 0.9 * absDiff.rows * absDiff.cols);

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
/*
TEST_F(TestChannelsExtractorGradMag, TestCompleteImageColor1OpenCL)
{
  cv::Mat image;
  image = cv::imread("images/index.jpeg", cv::IMREAD_COLOR);
  ASSERT_TRUE(image.data);
  compareGradientMagnitudeAndOrientation(image,
                                         "yaml/index_jpeg_GradientChannels.yaml",
                                         "opencl");
}

TEST_F(TestChannelsExtractorGradMag, TestCompleteImageGrayOpenCL)
{
  //cv::Mat mat1 = cv::Mat::zeros(1,1, CV_32FC1);
  //cv::Mat mat2 = cv::Mat::zeros(1,1, CV_32FC1);

  //mat1.at<float>(0,0) = 2;
  //mat2.at<float>(0,0) = 2;

  //at1.at<float>(0,1) = 28;
  //mat2.at<float>(0,1) = 11;

  //mat1.at<float>(1,0) = 24;
  //mat2.at<float>(1,0) = 23;

  //mat1.at<float>(1,1) = 1;
  //mat2.at<float>(1,1) = 1;
  //mat1.release();
  //mat2.release();
  //cv::Mat M0isMaximum = (mat1 & mat2);///255.0;
  //M0isMaximum.convertTo(M0isMaximum, CV_32F);
  //std::cout << M0isMaximum << std::endl;
  cv::ocl::Context context;
  if (!context.create(cv::ocl::Device::TYPE_GPU))
  {
    std::cout << "Failed creating the context..." << std::endl;
  }
  cv::ocl::Device(context.device(0));
  cv::ocl::setUseOpenCL(true);

  cv::Mat image;
  image = cv::imread("images/index.jpeg", cv::IMREAD_GRAYSCALE);
  ASSERT_TRUE(image.data);
  compareGradientMagnitudeAndOrientation(image,
                                         "yaml/index_jpeg_gray_GradientChannels.yaml",
                                         "opencl");
}*/

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

TEST_F(TestChannelsExtractorGradMag, TestGradMagGrayOpenCL)
{

  cv::FileStorage fs;
  bool file_exists = fs.open("yaml/index_jpeg_gray_GradientChannels.yaml", cv::FileStorage::READ);
  ASSERT_TRUE(file_exists);
  float normConst = readScalarFromFileNode(fs["normConst"]);
  float normRad = readScalarFromFileNode(fs["normRad"]);
  ChannelsExtractorGradMagOpenCL chanExtractOpenCL(normRad, normConst);

  std::vector<cv::UMat> gradMagExtractVector; 

  cv::Mat img0;
  img0 = cv::imread( "images/006733.png", cv::IMREAD_GRAYSCALE);

  img0.convertTo(img0, CV_32F);
  cv::UMat img, gray;
  img = img0.getUMat(cv::ACCESS_READ);

  cv::Mat splitted[3];
  split(img0, splitted);

  auto startLoad = std::chrono::system_clock::now();
  //for(int i = 0; i < 100; i++)
    gradMagExtractVector = chanExtractOpenCL.extractFeatures(img);
  
  auto endLoad = std::chrono::system_clock::now();
  std::chrono::duration<float,std::milli> durationLoad = endLoad - startLoad;
  std::cout << durationLoad.count() << "ms extractOpenCL " << std::endl;

  //cv::imshow("A",gradMagExtractVector[0]);
}



TEST_F(TestChannelsExtractorGradMag, TestGradMagGrayOpenCV)
{

  cv::FileStorage fs;
  bool file_exists = fs.open("yaml/index_jpeg_gray_GradientChannels.yaml", cv::FileStorage::READ);
  ASSERT_TRUE(file_exists);
  float normConst = readScalarFromFileNode(fs["normConst"]);
  float normRad = readScalarFromFileNode(fs["normRad"]);
  ChannelsExtractorGradMagOpenCV chanExtractOpenCV(normRad, normConst);

  std::vector<cv::Mat> gradMagExtractVector; 

  cv::Mat img2;
  img2 = cv::imread( "images/006733.png", cv::IMREAD_GRAYSCALE );//.getUMat(cv::ACCESS_READ);

  auto startLoad = std::chrono::system_clock::now();
  //for(int i = 0; i < 100; i++)
    gradMagExtractVector = chanExtractOpenCV.extractFeatures(img2);

  auto endLoad = std::chrono::system_clock::now();
  std::chrono::duration<float,std::milli> durationLoad = endLoad - startLoad;
  std::cout << durationLoad.count() << "ms extractOpenCV " << std::endl;

  //cv::imshow("B",gradMagExtractVector[0]);
  //cv::waitKey(0);
}



TEST_F(TestChannelsExtractorGradMag, TestGradMagColorOpenCL)
{

  cv::FileStorage fs;
  bool file_exists = fs.open("yaml/index_jpeg_gray_GradientChannels.yaml", cv::FileStorage::READ);
  ASSERT_TRUE(file_exists);
  float normConst = readScalarFromFileNode(fs["normConst"]);
  float normRad = readScalarFromFileNode(fs["normRad"]);
  ChannelsExtractorGradMagOpenCL chanExtractOpenCL(normRad, normConst);

  std::vector<cv::UMat> gradMagExtractVector; 

  cv::Mat img0;
  img0 = cv::imread( "images/006733.png", cv::IMREAD_COLOR);

  img0.convertTo(img0, CV_32F);
  cv::UMat img, gray;
  img = img0.getUMat(cv::ACCESS_READ);


  cv::ocl::Context context;
  if (!context.create(cv::ocl::Device::TYPE_GPU))
  {
    std::cout << "Failed creating the context..." << std::endl;
  }
  cv::ocl::Device(context.device(0));
  cv::ocl::setUseOpenCL(true);



  auto startLoad = std::chrono::system_clock::now();
  for(int i = 0; i < 10; i++)
    gradMagExtractVector = chanExtractOpenCL.extractFeatures(img);

  auto endLoad = std::chrono::system_clock::now();
  std::chrono::duration<float,std::milli> durationLoad = endLoad - startLoad;
  std::cout << durationLoad.count() << "ms extractOpenCL " << std::endl;


  ChannelsExtractorGradMagOpenCV chanExtractOpenCV(normRad, normConst);
  std::vector<cv::Mat> gradMagExtractVector2; 
  cv::Mat img2 = cv::imread( "images/006733.png", cv::IMREAD_COLOR );//.getUMat(cv::ACCESS_READ);
  auto startLoad2 = std::chrono::system_clock::now();

  for(int i = 0; i < 10; i++)
    gradMagExtractVector2 = chanExtractOpenCV.extractFeatures(img2);

  auto endLoad2 = std::chrono::system_clock::now();
  std::chrono::duration<float,std::milli> durationLoad2 = endLoad2 - startLoad2;
  std::cout << durationLoad2.count() << "ms extractOpenCV " << std::endl;

  //cv::UMat diff;
  //cv::subtract(gradMagExtractVector[0],gradMagExtractVector2[0],diff);
  cv::imshow("A",gradMagExtractVector[0]);

  cv::imshow("B",gradMagExtractVector2[0]);
  //cv::imshow("diff",diff);

  cv::waitKey(0);


}

/*
TEST_F(TestChannelsExtractorGradMag, TestGradMagColorOpenCV)
{

  cv::FileStorage fs;
  bool file_exists = fs.open("yaml/index_jpeg_gray_GradientChannels.yaml", cv::FileStorage::READ);
  ASSERT_TRUE(file_exists);
  float normConst = readScalarFromFileNode(fs["normConst"]);
  float normRad = readScalarFromFileNode(fs["normRad"]);
  ChannelsExtractorGradMagOpenCV chanExtractOpenCV(normRad, normConst);

  std::vector<cv::Mat> gradMagExtractVector; 

  cv::Mat img2;
  img2 = cv::imread( "images/006733.png", cv::IMREAD_GRAYSCALE );//.getUMat(cv::ACCESS_READ);

  auto startLoad = std::chrono::system_clock::now();
  gradMagExtractVector = chanExtractOpenCV.extractFeatures(img2);

  auto endLoad = std::chrono::system_clock::now();
  std::chrono::duration<float,std::milli> durationLoad = endLoad - startLoad;
  std::cout << durationLoad.count() << "ms extractOpenCV " << std::endl;

  cv::imshow("B",gradMagExtractVector[0]);
  cv::waitKey(0);
}
*/


