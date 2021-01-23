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
#include <channels/ChannelsExtractorGradHistOpenCL.h>
#include <channels/ChannelsExtractorGradMagOpenCV.h>
#include <channels/ChannelsExtractorGradMagOpenCL.h>

#include <channels/Utils.h>

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>

#include <iostream>
#include <chrono>


#undef DEBUG
//#define DEBUG

class TestChannelsExtractorGradHist: public testing::Test
{
 public:

  virtual void SetUp()
  {
    cv::ocl::Context context;
    if (!context.create(cv::ocl::Device::TYPE_GPU))
    {
      std::cout << "Failed creating OpenCL context..." << std::endl;
    }
    cv::ocl::Device(context.device(0));
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


  void compareOpenCLAndOpenCVSpeed
    (
    cv::Mat img,
    std::string matlab_grad_yaml_filename
    );

  void compareGradHistSyntheticImgOpenCvPDollar
    (
    cv::Mat gradMag,
    cv::Mat gradQuantizedOrient,
    float softBin,
    float binSize,
    int nOrients,
    float full
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
  // We use a common gradient magnitude extractor implementation: the P.Dollar's one that we know is working.
  pExtractorMag = dynamic_cast<ChannelsExtractorGradMag*>(new ChannelsExtractorGradMagPDollar(normRad, normConst));
  if (impl_type == "opencv")
  {
    pExtractorHist = dynamic_cast<ChannelsExtractorGradHist*>(new ChannelsExtractorGradHistOpenCV(binSize, nOrients, softBin, full));
  }  
   else if (impl_type == "opencl")
  {
    pExtractorHist = dynamic_cast<ChannelsExtractorGradHist*>(new ChannelsExtractorGradHistOpenCL(binSize, nOrients, softBin, full));
  }  

  else // "pdollar"
  {
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
    std::cout << "MatlabMat.size() = " << MatlabMat.size() << std::endl;
    std::cout << "gradHistExtractVector[i].size() = " << gradHistExtractVector[i].size() << std::endl;
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
    int num_pixels_ok = cv::sum(lessThanThr)[0];

#ifdef DEBUG
////    std::cout << "MatlabMat(cv::Range(0,8), cv::Range(0,8)) = " << std::endl;
////    std::cout << MatlabMat(cv::Range(0,8), cv::Range(0,8)) << std::endl;
////    std::cout << "gradHistExtractVector[i](cv::Range(0,8), cv::Range(0,8)) = " << std::endl;
////    std::cout << gradHistExtractVector[i](cv::Range(0,8), cv::Range(0,8))  << std::endl;
//    std::cout << "MatlabMat = " << std::endl;
//    std::cout << MatlabMat << std::endl;
//    std::cout << "gradHistExtractVector[i] = " << std::endl;
//    std::cout << gradHistExtractVector[i]  << std::endl;

//    std::cout << "num_pixels_ok = " << num_pixels_ok;
//    std::cout << " of " << absDiff.rows * absDiff.cols << std::endl;

//    cv::minMaxIdx(gradHistExtractVector[i], &min_val, &max_val, min_ind, max_ind, cv::Mat());
//    std::cout << "max value gradHist = " << max_val << std::endl;
//    std::cout << "min value gradHist = " << min_val << std::endl;

//    cv::minMaxIdx(MatlabMat, &min_val, &max_val, min_ind, max_ind, cv::Mat());
//    std::cout << "max value MatlabMat = " << max_val << std::endl;
//    std::cout << "min value MatlabMat = " << min_val << std::endl;

//    cv::minMaxIdx(absDiff, &min_val, &max_val, min_ind, max_ind, cv::Mat());
//    std::cout << "max value absDiff = " << max_val << std::endl;
//    std::cout << "min value absDiff = " << min_val << std::endl;

    cv::imshow("absDiff", absDiff);
    cv::imshow("cpp-Mat", gradHistExtractVector[i]);
    cv::imshow("matlab-Mat", MatlabMat);
    cv::waitKey();
#endif

    ASSERT_TRUE(num_pixels_ok > 0.7 * absDiff.rows * absDiff.cols);
  }
  fs.release();
}

void
TestChannelsExtractorGradHist::compareOpenCLAndOpenCVSpeed
  (
  cv::Mat img,
  std::string matlab_grad_yaml_filename
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
  ChannelsExtractorGradHist* pExtractorHistOpenCV;
  ChannelsExtractorGradHist* pExtractorHistOpenCL;
  // We use a common gradient magnitude extractor implementation: the P.Dollar's one that we know is working.
  pExtractorMag = dynamic_cast<ChannelsExtractorGradMag*>(new ChannelsExtractorGradMagPDollar(normRad, normConst));
  pExtractorHistOpenCV = dynamic_cast<ChannelsExtractorGradHist*>(new ChannelsExtractorGradHistOpenCV(binSize, nOrients, softBin, full));
  pExtractorHistOpenCL = dynamic_cast<ChannelsExtractorGradHist*>(new ChannelsExtractorGradHistOpenCL(binSize, nOrients, softBin, full));

  // Extract the gradient histogram channels
  std::vector<cv::Mat> gradMagExtractVector;
  std::vector<cv::Mat> gradHistExtractVector;
  gradMagExtractVector = pExtractorMag->extractFeatures(img);

  // Transparent API tests -------------------------------------------------
  cv::ocl::setUseOpenCL(true);
  for (int i=0; i<10; i++)
  {
    auto startLoad = std::chrono::system_clock::now();
    gradHistExtractVector = pExtractorHistOpenCL->extractFeatures(img, gradMagExtractVector);
    auto endLoad = std::chrono::system_clock::now();
    std::chrono::duration<float,std::milli> durationLoad = endLoad - startLoad;
    std::cout << durationLoad.count() << "ms gradHistOpenCL " << std::endl;
  }

  std::cout << "---------------" << std::endl;

  // OpenCV tests ----------------------------------------------------------
  cv::ocl::setUseOpenCL(false);
  for (int i=0; i<10; i++)
  {
    auto startLoad = std::chrono::system_clock::now();
    gradHistExtractVector = pExtractorHistOpenCV->extractFeatures(img, gradMagExtractVector);
    auto endLoad = std::chrono::system_clock::now();
    std::chrono::duration<float,std::milli> durationLoad = endLoad - startLoad;
    std::cout << durationLoad.count() << "ms gradHistOpenCV " << std::endl;
  }

  delete pExtractorMag;
  delete pExtractorHistOpenCV;
  delete pExtractorHistOpenCL;

  fs.release();
}

//-------------------------------------------------------------------
//
// P.Dollar implementation tests
//
//-------------------------------------------------------------------
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



//-------------------------------------------------------------------
//
// OpenCV cv::Mat based implementation tests
//
//-------------------------------------------------------------------
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

//-------------------------------------------------------------------
//
// OpenCV cv::UMat based implementation tests (OpenCL).
//
//-------------------------------------------------------------------
TEST_F(TestChannelsExtractorGradHist, TestCompleteImageColor1OpenCL)
{
  cv::Mat image;
  image = cv::imread("images/index.jpeg", cv::IMREAD_COLOR);
  ASSERT_TRUE(image.data);
  compareGradientOrientationHistogram(image,
                                      "yaml/index_jpeg_GradientChannels.yaml",
                                      "opencl");

  image = cv::imread("images/coches3.jpg", cv::IMREAD_COLOR);
  ASSERT_TRUE(image.data);
  compareOpenCLAndOpenCVSpeed(image,
                              "yaml/index_jpeg_GradientChannels.yaml");
}

TEST_F(TestChannelsExtractorGradHist, TestCompleteImageGrayOpenCL)
{
  cv::Mat image;
  image = cv::imread("images/index.jpeg", cv::IMREAD_GRAYSCALE);
  ASSERT_TRUE(image.data);
  compareGradientOrientationHistogram(image,
                                      "yaml/index_jpeg_gray_GradientChannels.yaml",
                                      "opencl");

  image = cv::imread("images/coches3.jpg", cv::IMREAD_COLOR);
  ASSERT_TRUE(image.data);
  compareOpenCLAndOpenCVSpeed(image,
                              "yaml/index_jpeg_gray_GradientChannels.yaml");
}

TEST_F(TestChannelsExtractorGradHist, TestCompleteImageColorSoftBinNeg4OpenCL)
{
  cv::Mat image;
  image = cv::imread("images/index.jpeg", cv::IMREAD_COLOR);
  ASSERT_TRUE(image.data);
  compareGradientOrientationHistogram(image,
                                      "yaml/index_jpeg_GradientChannels_softBin_negative_4.yaml",
                                      "opencl");
}


TEST_F(TestChannelsExtractorGradHist, TestCompleteImageColorBinSize5OpenCL)
{
  cv::Mat image;
  image = cv::imread("images/index.jpeg", cv::IMREAD_COLOR);
  ASSERT_TRUE(image.data);
  compareGradientOrientationHistogram(image,
                                      "yaml/index_jpeg_GradientChannels_binSize_5.yaml",
                                      "opencl");
}

TEST_F(TestChannelsExtractorGradHist, TestCompleteImageColorBinSize1SoftBin2OpenCL)
{
  cv::Mat image;
  image = cv::imread("images/index.jpeg", cv::IMREAD_COLOR);
  ASSERT_TRUE(image.data);
  compareGradientOrientationHistogram(image,
                                      "yaml/index_jpeg_GradientChannels_binSize_1_softBin_2.yaml",
                                      "opencl");
}

TEST_F(TestChannelsExtractorGradHist, TestCompleteImageColorNormConst0_07OpenCL)
{
  cv::Mat image;
  image = cv::imread("images/index.jpeg", cv::IMREAD_COLOR);
  ASSERT_TRUE(image.data);
  compareGradientOrientationHistogram(image,
                                      "yaml/index_jpeg_GradientChannels_normConst_0_07.yaml",
                                      "opencl");
}

TEST_F(TestChannelsExtractorGradHist, TestCompleteImageColorNormRad0OpenCL)
{
  cv::Mat image;
  image = cv::imread("images/index.jpeg", cv::IMREAD_COLOR);
  ASSERT_TRUE(image.data);
  compareGradientOrientationHistogram(image,
                                         "yaml/index_jpeg_GradientChannels_normRad_0.yaml",
                                         "opencl");
}


void
TestChannelsExtractorGradHist::compareGradHistSyntheticImgOpenCvPDollar
  (
  cv::Mat M,
  cv::Mat O,
  float softBin,
  float binSize,
  int nOrients,
  float full
  )
{
  ChannelsExtractorGradHist* pExtractorHistOpenCV;
  ChannelsExtractorGradHist* pExtractorHistPDollar;
  pExtractorHistOpenCV = dynamic_cast<ChannelsExtractorGradHist*>(new ChannelsExtractorGradHistOpenCV(binSize, nOrients, softBin, full));
  pExtractorHistPDollar = dynamic_cast<ChannelsExtractorGradHist*>(new ChannelsExtractorGradHistPDollar(binSize, nOrients, softBin, full));

  // Extract the gradient histogram channels
  std::vector<cv::Mat> gradMagExtractVector(2);
  gradMagExtractVector[0] = M;
  gradMagExtractVector[1] = O;
  cv::Mat img = cv::Mat::ones(M.rows, M.cols, CV_32F);
  std::vector<cv::Mat> gradHistExtractVectorOpenCV;
  std::vector<cv::Mat> gradHistExtractVectorPDollar;
  gradHistExtractVectorOpenCV = pExtractorHistOpenCV->extractFeatures(img, gradMagExtractVector);
  gradHistExtractVectorPDollar = pExtractorHistPDollar->extractFeatures(img, gradMagExtractVector);
  delete pExtractorHistOpenCV;
  delete pExtractorHistPDollar;

  // Compare OpenCV gradient histograms with PDollar implementation
  for (uint i=0; i < 1; i++) //gradHistExtractVectorOpenCV.size(); i++)
  {
    double min_val;
    double max_val;
    int min_ind[2];
    int max_ind[2];
    cv::minMaxIdx(gradHistExtractVectorPDollar[i], &min_val, &max_val, min_ind, max_ind, cv::Mat());
    float channel_range = max_val - min_val;    
    float threshold = 0.05*channel_range; // Asume a 5% of the channel's range is an acceptable error.
    if (threshold == 0.0)
    {
      threshold = 0.1;
    }
    cv::Mat absDiff = cv::abs(gradHistExtractVectorOpenCV[i] - gradHistExtractVectorPDollar[i]);
    cv::Mat lessThanThr = (absDiff < threshold)/255.0; // Boolean matrix has 255 for true and 0 for false.
    int num_pixels_ok = cv::sum(lessThanThr)[0];

#ifdef DEBUG
    std::cout << "softBin = " << softBin << std::endl;
    std::cout << "binSize = " << binSize << std::endl;
    std::cout << "gradHistExtractVectorPDollar[i] = " << std::endl;
    std::cout << gradHistExtractVectorPDollar[i] << std::endl;
    std::cout << "gradHistExtractVectorOpenCV[i] = " << std::endl;
    std::cout << gradHistExtractVectorOpenCV[i]  << std::endl;

    std::cout << "num_pixels_ok = " << num_pixels_ok;
    std::cout << " of " << absDiff.rows * absDiff.cols << std::endl;

//    cv::imshow("absDiff", absDiff);
//    cv::imshow("opencv-Mat", gradHistExtractVectorOpenCV[i]);
//    cv::imshow("pdollar-Mat", gradHistExtractVectorPDollar[i]);
//    cv::waitKey();
#endif

    // Keep the 0.7 as the pixels in the borders of the image do not get the values as in P.Dollar. In this
    // case the images are small and the border is quite a big proportion of them.
    ASSERT_TRUE(num_pixels_ok > 0.7 * absDiff.rows * absDiff.cols);
  }
}

TEST_F(TestChannelsExtractorGradHist, TestCompareOpenCvPdollarGradHistSequentialMagSoftBin1binSize2)
{
  cv::Mat gradMag = cv::Mat::ones(40, 60, CV_32F);
  cv::Mat gradQuantizedOrient = cv::Mat::zeros(40, 60, CV_32F);

  float k = 0.0;
  for (int i=0; i < gradMag.rows; i++)
  {
    for (int j=0; j < gradMag.cols; j++)
    {
      gradMag.at<float>(i,j) = k;
      k += 1.0;
    }
  }

  compareGradHistSyntheticImgOpenCvPDollar(
    gradMag,
    gradQuantizedOrient,
    1, //softBin,
    2,  // binSize,
    6,  // nOrients,
    0); // full
}

TEST_F(TestChannelsExtractorGradHist, TestCompareOpenCvPdollarGradHistSequentialMagSoftBin1binSize3)
{
  cv::Mat gradMag = cv::Mat::ones(43, 61, CV_32F);
  cv::Mat gradQuantizedOrient = cv::Mat::zeros(43, 61, CV_32F);

  float k = 0.0;
  for (int i=0; i < gradMag.rows; i++)
  {
    for (int j=0; j < gradMag.cols; j++)
    {
      gradMag.at<float>(i,j) = k;
      k += 1.0;
    }
  }

  compareGradHistSyntheticImgOpenCvPDollar(
    gradMag,
    gradQuantizedOrient,
    1, //softBin,
    3,  // binSize,
    6,  // nOrients,
    0); // full
}

TEST_F(TestChannelsExtractorGradHist, TestCompareOpenCvPdollarGradHistSequentialMagSoftBin1binSize5)
{
  cv::Mat gradMag = cv::Mat::ones(43, 61, CV_32F);
  cv::Mat gradQuantizedOrient = cv::Mat::zeros(43, 61, CV_32F);

  float k = 0.0;
  for (int i=0; i < gradMag.rows; i++)
  {
    for (int j=0; j < gradMag.cols; j++)
    {
      gradMag.at<float>(i,j) = k;
      k += 1.0;
    }
  }

  compareGradHistSyntheticImgOpenCvPDollar(
    gradMag,
    gradQuantizedOrient,
    1, //softBin,
    5,  // binSize,
    6,  // nOrients,
    0); // full
}

TEST_F(TestChannelsExtractorGradHist, TestCompareOpenCvPdollarGradHistSequentialMagSoftBinMinus2binSize1)
{
  cv::Mat gradMag = cv::Mat::ones(43, 61, CV_32F);
  cv::Mat gradQuantizedOrient = cv::Mat::zeros(43, 61, CV_32F);

  float k = 0.0;
  for (int i=0; i < gradMag.rows; i++)
  {
    for (int j=0; j < gradMag.cols; j++)
    {
      gradMag.at<float>(i,j) = k;
      k += 1.0;
    }
  }

  compareGradHistSyntheticImgOpenCvPDollar(
    gradMag,
    gradQuantizedOrient,
    -2, //softBin,
    1,  // binSize,
    6,  // nOrients,
    0); // full
}

TEST_F(TestChannelsExtractorGradHist, TestCompareOpenCvPdollarGradHistOnesSoftBin1binSize2)
{
  cv::Mat gradMag = cv::Mat::ones(43, 61, CV_32F);
  cv::Mat gradQuantizedOrient = cv::Mat::zeros(43, 61, CV_32F);

  float k = 0.0;
  for (int i=0; i < gradMag.rows; i++)
  {
    for (int j=0; j < gradMag.cols; j++)
    {
      gradMag.at<float>(i,j) = k;
      k += 1.0;
    }
  }

  compareGradHistSyntheticImgOpenCvPDollar(
    gradMag,
    gradQuantizedOrient,
    1, //softBin,
    2,  // binSize,
    6,  // nOrients,
    0); // full
}

TEST_F(TestChannelsExtractorGradHist, TestCompareOpenCvPdollarGradHistOnesSoftBinMinus4binSize5)
{
  cv::Mat gradMag = cv::Mat::ones(43, 61, CV_32F);
  cv::Mat gradQuantizedOrient = cv::Mat::zeros(43, 61, CV_32F);

  float k = 0.0;
  for (int i=0; i < gradMag.rows; i++)
  {
    for (int j=0; j < gradMag.cols; j++)
    {
      gradMag.at<float>(i,j) = k;
      k += 1.0;
    }
  }

  compareGradHistSyntheticImgOpenCvPDollar(
    gradMag,
    gradQuantizedOrient,
    -2, //softBin,
    5,  // binSize,
    6,  // nOrients,
    0); // full
}

TEST_F(TestChannelsExtractorGradHist, TestCompareOpenCvPdollarGradHistOnesSoftBinMinus4binSize3)
{
  cv::Mat gradMag = cv::Mat::ones(43, 61, CV_32F);
  cv::Mat gradQuantizedOrient = cv::Mat::zeros(43, 61, CV_32F);

  float k = 0.0;
  for (int i=0; i < gradMag.rows; i++)
  {
    for (int j=0; j < gradMag.cols; j++)
    {
      gradMag.at<float>(i,j) = k;
      k += 1.0;
    }
  }

  compareGradHistSyntheticImgOpenCvPDollar(
    gradMag,
    gradQuantizedOrient,
    -2, //softBin,
    3,  // binSize,
    6,  // nOrients,
    0); // full
}

TEST_F(TestChannelsExtractorGradHist, TestCompareOpenCvPdollarGradHistOnesSoftBin2binSize1)
{
  cv::Mat gradMag = cv::Mat::ones(43, 61, CV_32F);
  cv::Mat gradQuantizedOrient = cv::Mat::zeros(43, 61, CV_32F);

  float k = 0.0;
  for (int i=0; i < gradMag.rows; i++)
  {
    for (int j=0; j < gradMag.cols; j++)
    {
      gradMag.at<float>(i,j) = k;
      k += 1.0;
    }
  }

  compareGradHistSyntheticImgOpenCvPDollar(
    gradMag,
    gradQuantizedOrient,
    2, //softBin,
    1,  // binSize,
    6,  // nOrients,
    0); // full
}

//------- Random synthetic Orientations
TEST_F(TestChannelsExtractorGradHist, TestCompareOpenCvPdollarGradHistRandomOrientationsSequentialMagSoftBin2binSize2)
{
  cv::Mat gradMag = cv::Mat::ones(43, 61, CV_32F);
  cv::Mat gradQuantizedOrient = cv::Mat::zeros(43, 61, CV_32S);

  float low = 0;
  float high = M_PI;
  cv::randu(gradQuantizedOrient, cv::Scalar(low), cv::Scalar(high));
  gradQuantizedOrient.convertTo(gradQuantizedOrient, CV_32F);

  float k = 0.0;
  for (int i=0; i < gradMag.rows; i++)
  {
    for (int j=0; j < gradMag.cols; j++)
    {
      gradMag.at<float>(i,j) = k;
      k += 1.0;
    }
  }

  compareGradHistSyntheticImgOpenCvPDollar(
    gradMag,
    gradQuantizedOrient,
    2, //softBin,
    2,  // binSize,
    6,  // nOrients,
    0); // full
}


TEST_F(TestChannelsExtractorGradHist, TestCompareOpenCvPdollarGradHistRandomOrientationsSequentialMagSoftBin1binSize3)
{
  cv::Mat gradMag = cv::Mat::ones(43, 61, CV_32F);
  cv::Mat gradQuantizedOrient = cv::Mat::zeros(43, 61, CV_32F);

  float low = 0;
  float high = M_PI;
  cv::randu(gradQuantizedOrient, cv::Scalar(low), cv::Scalar(high));

  float k = 0.0;
  for (int i=0; i < gradMag.rows; i++)
  {
    for (int j=0; j < gradMag.cols; j++)
    {
      gradMag.at<float>(i,j) = k;
      k += 1.0;
    }
  }

  compareGradHistSyntheticImgOpenCvPDollar(
    gradMag,
    gradQuantizedOrient,
    1, //softBin,
    3,  // binSize,
    6,  // nOrients,
    0); // full
}


TEST_F(TestChannelsExtractorGradHist, TestCompareOpenCvPdollarGradHistRandomOrientationsSequentialMagSoftBin1binSize5)
{
  cv::Mat gradMag = cv::Mat::ones(43, 61, CV_32F);
  cv::Mat gradQuantizedOrient = cv::Mat::zeros(43, 61, CV_32F);

  float low = 0;
  float high = M_PI;
  cv::randu(gradQuantizedOrient, cv::Scalar(low), cv::Scalar(high));

  float k = 0.0;
  for (int i=0; i < gradMag.rows; i++)
  {
    for (int j=0; j < gradMag.cols; j++)
    {
      gradMag.at<float>(i,j) = k;
      k += 1.0;
    }
  }

  compareGradHistSyntheticImgOpenCvPDollar(
    gradMag,
    gradQuantizedOrient,
    1, //softBin,
    5,  // binSize,
    6,  // nOrients,
    0); // full
}

//TEST_F(TestChannelsExtractorGradHist, TestCompareFloats)
//{
//    /*cv::ocl::Context context;
//    if (!context.create(cv::ocl::Device::TYPE_GPU))
//    {
//        std::cout << "Failed creating the context..." << std::endl;
//    }
//    // In OpenCV 3.0.0 beta, only a single device is detected.
//    std::cout << context.ndevices() << " GPU devices are detected." << std::endl;
//    for (int i = 0; i < context.ndevices(); i++)
//    {
//        cv::ocl::Device device = context.device(i);
//        std::cout << "name                 : " << device.name() << std::endl;
//        std::cout << "available            : " << device.available() << std::endl;
//        std::cout << "imageSupport         : " << device.imageSupport() << std::endl;
//        std::cout << "OpenCL_C_Version     : " << device.OpenCL_C_Version() << std::endl;
//        std::cout << std::endl;
//    }
//    /*cout << "OpenCV version : " << CV_VERSION << endl;
//    cout << "Major version : " << CV_MAJOR_VERSION << endl;
//    cout << "Minor version : " << CV_MINOR_VERSION << endl;
//    cout << "Subminor version : " << CV_SUBMINOR_VERSION << endl;* /
//    // Select the first device
//    cv::ocl::Device(context.device(0));
//    cv::ocl::setUseOpenCL(true); */
//    // Transfer Mat data to the device
//    /*cv::Mat mat_src = cv::imread("006733.png", cv::IMREAD_GRAYSCALE);
//    mat_src.convertTo(mat_src, CV_32F, 1.0 / 255);
//    cv::UMat umat_src = mat_src.getUMat(cv::ACCESS_READ, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
//    cv::UMat umat_dst(mat_src.size(), CV_32F, cv::ACCESS_WRITE, cv::USAGE_ALLOCATE_DEVICE_MEMORY);

//    cv::Mat mat_dst = umat_dst.getMat(cv::ACCESS_READ);

//    cv::imshow("src", mat_src);
//    cv::waitKey(0);
//    std::cout << "FINALIZA" << std::endl;*/
////    cv::UMat img, gray;
//    cv::Mat img, gray;
////    img = cv::imread( "images/006733.png", cv::IMREAD_COLOR  ).getUMat( cv::ACCESS_READ);
//    img = cv::imread( "images/coches3.jpg", cv::IMREAD_COLOR  );
//    cv::UMat img_UMat, gray_UMat;
//    img.convertTo(img_UMat, CV_8UC1);
//    //cv::Mat img, gray;
//    //img = cv::imread( "images/006733.png", cv::IMREAD_COLOR );//.getUMat(cv::ACCESS_READ);
//    //img.convertTo(img, CV_32FC1);
//    //cv::UMat img, gray;
//    //img = img2.getUMat(cv::ACCESS_READ, cv::USAGE_ALLOCATE_DEVICE_MEMORY);

//    for(int i = 0; i < 10; i++)
//    {
//        auto startLoad = std::chrono::system_clock::now();
//        cv::UMat img_UMat, gray_UMat;
//        img.convertTo(img_UMat, CV_8UC1);
////        img.convertTo(img_UMat, CV_32F);
//        cv::cvtColor(img_UMat, gray_UMat, cv::COLOR_BGR2GRAY);
//        cv::GaussianBlur(gray_UMat, gray_UMat, cv::Size(7, 7), 1.5);
//        //Canny(gray_UMat, gray_UMat, 0, 50);
//        gray_UMat.copyTo(gray);
//        auto endLoad = std::chrono::system_clock::now();
//        std::chrono::duration<float,std::milli> durationLoad = endLoad - startLoad;
//        std::cout << durationLoad.count() << "ms " << std::endl;
//    }

//}

/*
TEST_F(TestChannelsExtractorGradHist, TestOpenCLImage)
{
  cv::FileStorage fs;
  bool file_exists = fs.open("yaml/index_jpeg_gray_GradientChannels.yaml", cv::FileStorage::READ);
  ASSERT_TRUE(file_exists);
  float normConst = readScalarFromFileNode(fs["normConst"]);
  float normRad = readScalarFromFileNode(fs["normRad"]);


  file_exists = fs.open("yaml/index_jpeg_GradientChannels.yaml", cv::FileStorage::READ);
  ASSERT_TRUE(file_exists);

  float nOrients = readScalarFromFileNode(fs["nOrients"]);
  float binSize = readScalarFromFileNode(fs["binSize"]);
  float softBin = readScalarFromFileNode(fs["softBin"]);
  float full = readScalarFromFileNode(fs["full"]);


  cv::Mat image;
  image = cv::imread("images/index.jpeg", cv::IMREAD_COLOR);

  cv::Mat img_float;
  image.convertTo(img_float, CV_32FC1); // important to have continuous memory in img_aux.ptr<float>
  cv::UMat IM = img_float.getUMat(cv::ACCESS_READ);

  ChannelsExtractorGradMagOpenCL chanExtractOpenCL(normRad, normConst);
  std::vector<cv::UMat> gradMagExtractVector; 

  gradMagExtractVector = chanExtractOpenCL.extractFeatures(IM);

  std::vector<cv::UMat> gradHistExtractVector;
  ChannelsExtractorGradHistOpenCL ChannelsExtractrGradHistOpenCL(binSize, nOrients, softBin, full);

  auto startLoad = std::chrono::system_clock::now();
  for(int i = 0; i < 50; i++)
    gradHistExtractVector = ChannelsExtractrGradHistOpenCL.extractFeatures(IM, gradMagExtractVector);

  auto endLoad = std::chrono::system_clock::now();
  std::chrono::duration<float,std::milli> durationLoad = endLoad - startLoad;
  std::cout << durationLoad.count() << "ms gradHistOpenCL " << std::endl;

  std::vector<cv::Mat> gradHistExtractVectorOpenCV;

  ChannelsExtractorGradHistOpenCV ChannelsExtractrGradHistOpenCV(binSize, nOrients, softBin, full);

  std::vector<cv::Mat> gradMagExtractOpenCV;
  gradMagExtractOpenCV.push_back(gradMagExtractVector[0].getMat(cv::ACCESS_READ));
  gradMagExtractOpenCV.push_back(gradMagExtractVector[1].getMat(cv::ACCESS_READ));

  startLoad = std::chrono::system_clock::now();
  
  for(int i = 0; i < 50; i++)
    gradHistExtractVectorOpenCV = ChannelsExtractrGradHistOpenCV.extractFeatures(image, gradMagExtractOpenCV);

  endLoad = std::chrono::system_clock::now();
  durationLoad = endLoad - startLoad;
  std::cout << durationLoad.count() << "ms gradHistOpenCL " << std::endl;


  for(int i = 0; i < 6; i++)
  {
    cv::Mat diff =  gradHistExtractVectorOpenCV[i] - gradHistExtractVector[i].getMat(cv::ACCESS_READ);
    double minVal; double maxVal; 
    minMaxLoc( diff, &minVal, &maxVal);

    ASSERT_TRUE(minVal < 0.0001);
    ASSERT_TRUE(maxVal < 0.0001);
  }
  //std::cout << minVal << maxVal << std::endl;

}
*/



