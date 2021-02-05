/** ------------------------------------------------------------------------
 *
 *  @brief Test of Channels Extractor for LUV color space
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2019/07/08
 *
 *  ------------------------------------------------------------------------ */

#include <channels/ChannelsExtractorLUV.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <gtest/gtest.h>

#undef VISUALIZE_RESULTS
//#define VISUALIZE_RESULTS
//#define DEBUG
#undef DEBUG

class TestChannelsExtractorLUV: public testing::Test
{
public:
  const int ROWS = 5;
  const int COLS = 5;

  std::shared_ptr<ChannelsExtractorLUV> channExtractorPDollar;
  std::shared_ptr<ChannelsExtractorLUV> channExtractorSmoothPDollar;
  std::shared_ptr<ChannelsExtractorLUV> channExtractorOpenCV;
  std::shared_ptr<ChannelsExtractorLUV> channExtractorSmoothOpenCV;
  std::shared_ptr<ChannelsExtractorLUV> channExtractorOpenCL;
  std::shared_ptr<ChannelsExtractorLUV> channExtractorSmoothOpenCL;

  virtual void SetUp()
  {
    cv::ocl::Context context;
    if (!context.create(cv::ocl::Device::TYPE_GPU))
    {
      std::cout << "Failed creating OpenCL context..." << std::endl;
    }
    cv::ocl::Device(context.device(0));
    cv::ocl::setUseOpenCL(true);

    channExtractorPDollar = ChannelsExtractorLUV::createExtractor("pdollar", false, 1);
    channExtractorSmoothPDollar = ChannelsExtractorLUV::createExtractor("pdollar", true, 5);
    channExtractorOpenCV = ChannelsExtractorLUV::createExtractor("opencv", false, 1);
    channExtractorSmoothOpenCV = ChannelsExtractorLUV::createExtractor("opencv", true, 5);
    channExtractorOpenCL = ChannelsExtractorLUV::createExtractor("opencl", false, 1);
    channExtractorSmoothOpenCL = ChannelsExtractorLUV::createExtractor("opencl", true, 5);
  }

  virtual void TearDown()
  {
  }

  void
  compareLUVWithPDollar
    (
    cv::Mat img,
    std::shared_ptr<ChannelsExtractorLUV>& pExtractorPDollar,
    std::shared_ptr<ChannelsExtractorLUV>& pExtractor
    );
};

// -------------------------------------------------------------------------------
// -------------------------------------------------------------------------------
// compareLUVWithPDollar
// -------------------------------------------------------------------------------
// -------------------------------------------------------------------------------
void
TestChannelsExtractorLUV::compareLUVWithPDollar
  (
  cv::Mat img,
  std::shared_ptr<ChannelsExtractorLUV>& pExtractorPDollar,
  std::shared_ptr<ChannelsExtractorLUV>& pExtractor
  )
{
  // Extract the gradient magnitude and orientation channels
  std::vector<cv::Mat> luvExtractVector = pExtractor->extractFeatures(img);
  std::vector<cv::Mat> luvExtractVectorPDollar = pExtractorPDollar->extractFeatures(img);

  // Compare P.Dollar LUV with C++ implementation
  for(uint i=0; i<3; i++)
  {
    double min_val;
    double max_val;
    int min_ind[2];
    int max_ind[2];
    cv::minMaxIdx(luvExtractVectorPDollar[i], &min_val, &max_val, min_ind, max_ind, cv::Mat());
    float channel_range = max_val - min_val;   
    float threshold;
    if (channel_range == 0.0)
    {
      threshold = 0.05;
    }
    else
    {
      threshold = 0.1*channel_range; // Asume a 10% of the maximum value is an acceptable error.
    }
    cv::Mat absDiff = cv::abs(luvExtractVector[i] - luvExtractVectorPDollar[i]);
    cv::Mat lessThanThr = (absDiff < threshold)/255.0; // Boolean matrix has 255 for true and 0 for false.
    int num_pixels_ok = cv::sum(lessThanThr)[0];

#ifdef DEBUG
    std::cout << "num_pixels_ok = " << num_pixels_ok;
    std::cout << " of " << absDiff.rows * absDiff.cols << std::endl;
    std::cout << "luvExtractVectorPDollar[i]=" << luvExtractVectorPDollar[i](cv::Range(0,5), cv::Range(0,5)) << std::endl;
    std::cout << "luvExtractVector[i]=" <<  luvExtractVector[i](cv::Range(0,5), cv::Range(0,5)) << std::endl;
    std::cout << "absDiff=" <<  absDiff(cv::Range(0,5), cv::Range(0,5)) << std::endl;
    std::cout << "lessThanThr=" <<  lessThanThr(cv::Range(0,5), cv::Range(0,5)) << std::endl;
//    cv::imshow("absDiff", absDiff);
//    cv::imshow("cpp-LUV", luvExtractVector[i]);
//    cv::imshow("PDollar-LUV", luvExtractVectorPDollar[i]);
//    cv::waitKey();
    std::cout << "====================== " << std::endl;
#endif

    ASSERT_TRUE(num_pixels_ok > 0.9 * absDiff.size().width * absDiff.size().height);
  }
}


// -------------------------------------------------------------------------------
// -------------------------------------------------------------------------------
// Test LUV OpenCV.
// -------------------------------------------------------------------------------
// -------------------------------------------------------------------------------

TEST_F(TestChannelsExtractorLUV, TestWhiteImageOpenCV)
{
  // Setting white image (255,255, 255) in all pixels .
  cv::Mat img(cv::Size(COLS, ROWS), CV_8UC3, cv::Scalar(255, 255, 255));

  compareLUVWithPDollar(img, channExtractorPDollar, channExtractorOpenCV);
  compareLUVWithPDollar(img, channExtractorSmoothPDollar, channExtractorSmoothOpenCV);
}

TEST_F(TestChannelsExtractorLUV, TestBlackImageOpenCV)
{
  // Setting black image (0,0,0) in all pixels.
  cv::Mat img(cv::Size(COLS, ROWS), CV_8UC3, cv::Scalar(0,0,0));

  compareLUVWithPDollar(img, channExtractorPDollar, channExtractorOpenCV);
  compareLUVWithPDollar(img, channExtractorSmoothPDollar, channExtractorSmoothOpenCV);
}

//TEST_F(TestChannelsExtractorLUV, TestMediumGrayImageOpenCV)
//{
//  // Setting medium gray image (128, 128, 128) in all pixels .
//  cv::Mat img(cv::Size(COLS, ROWS), CV_8UC3, cv::Scalar(128, 128, 128));

//  compareLUVWithPDollar(img, channExtractorPDollar, channExtractorOpenCV);
//  compareLUVWithPDollar(img, channExtractorSmoothPDollar, channExtractorSmoothOpenCV);
//}

TEST_F(TestChannelsExtractorLUV, TestRedImageOpenCV)
{
  cv::Mat img(cv::Size(COLS, ROWS), CV_8UC3, cv::Scalar(0, 0, 255)); // BRG

  compareLUVWithPDollar(img, channExtractorPDollar, channExtractorOpenCV);
  compareLUVWithPDollar(img, channExtractorSmoothPDollar, channExtractorSmoothOpenCV);
}

TEST_F(TestChannelsExtractorLUV, TestBlueImageOpenCV)
{
  cv::Mat img(cv::Size(COLS, ROWS), CV_8UC3, cv::Scalar(255, 0, 0)); // BRG

  compareLUVWithPDollar(img, channExtractorPDollar, channExtractorOpenCV);
  compareLUVWithPDollar(img, channExtractorSmoothPDollar, channExtractorSmoothOpenCV);
}

TEST_F(TestChannelsExtractorLUV, TestGreenImageOpenCV)
{
  cv::Mat img(cv::Size(COLS, ROWS), CV_8UC3, cv::Scalar(0, 255, 0)); // BGR

  compareLUVWithPDollar(img, channExtractorPDollar, channExtractorOpenCV);
  compareLUVWithPDollar(img, channExtractorSmoothPDollar, channExtractorSmoothOpenCV);
}

//TEST_F(TestChannelsExtractorLUV, TestNaturalRGBImageOpenCV)
//{
//  cv::Mat img = cv::imread("images/index.jpeg");

//  compareLUVWithPDollar(img, channExtractorPDollar, channExtractorOpenCV);
//  compareLUVWithPDollar(img, channExtractorSmoothPDollar, channExtractorSmoothOpenCV);
//}

//TEST_F(TestChannelsExtractorLUV, TestNaturalSmoothRGBImageOpenCV)
//{
//  cv::Mat img = cv::imread("images/index.jpeg");

//  compareLUVWithPDollar(img, channExtractorPDollar, channExtractorOpenCV);
//  compareLUVWithPDollar(img, channExtractorSmoothPDollar, channExtractorSmoothOpenCV);
//}

// -------------------------------------------------------------------------------
// -------------------------------------------------------------------------------
// Test LUV OpenCL.
// -------------------------------------------------------------------------------
// -------------------------------------------------------------------------------

TEST_F(TestChannelsExtractorLUV, TestWhiteImageOpenCL)
{
  // Setting white image (255,255, 255) in all pixels .
  cv::Mat img(cv::Size(COLS, ROWS), CV_8UC3, cv::Scalar(255, 255, 255));

  compareLUVWithPDollar(img, channExtractorPDollar, channExtractorOpenCL);
  compareLUVWithPDollar(img, channExtractorSmoothPDollar, channExtractorSmoothOpenCL);
}

TEST_F(TestChannelsExtractorLUV, TestBlackImageOpenCL)
{
  // Setting black image (0,0,0) in all pixels.
  cv::Mat img(cv::Size(COLS, ROWS), CV_8UC3, cv::Scalar(0,0,0));

  compareLUVWithPDollar(img, channExtractorPDollar, channExtractorOpenCL);
  compareLUVWithPDollar(img, channExtractorSmoothPDollar, channExtractorSmoothOpenCL);
}

//TEST_F(TestChannelsExtractorLUV, TestMediumGrayImageOpenCL)
//{
//  // Setting medium gray image (128, 128, 128) in all pixels .
//  cv::Mat img(cv::Size(COLS, ROWS), CV_8UC3, cv::Scalar(128, 128, 128));

//  compareLUVWithPDollar(img, channExtractorPDollar, channExtractorOpenCL);
//  compareLUVWithPDollar(img, channExtractorSmoothPDollar, channExtractorSmoothOpenCL);
//}

TEST_F(TestChannelsExtractorLUV, TestRedImageOpenCL)
{
  cv::Mat img(cv::Size(COLS, ROWS), CV_8UC3, cv::Scalar(0, 0, 255)); // BRG

  compareLUVWithPDollar(img, channExtractorPDollar, channExtractorOpenCL);
  compareLUVWithPDollar(img, channExtractorSmoothPDollar, channExtractorSmoothOpenCL);
}

TEST_F(TestChannelsExtractorLUV, TestBlueImageOpenCL)
{
  cv::Mat img(cv::Size(COLS, ROWS), CV_8UC3, cv::Scalar(255, 0, 0)); // BRG

  compareLUVWithPDollar(img, channExtractorPDollar, channExtractorOpenCL);
  compareLUVWithPDollar(img, channExtractorSmoothPDollar, channExtractorSmoothOpenCL);
}

TEST_F(TestChannelsExtractorLUV, TestGreenImageOpenCL)
{
  cv::Mat img(cv::Size(COLS, ROWS), CV_8UC3, cv::Scalar(0, 255, 0)); // BGR

  compareLUVWithPDollar(img, channExtractorPDollar, channExtractorOpenCL);
  compareLUVWithPDollar(img, channExtractorSmoothPDollar, channExtractorSmoothOpenCL);
}

//TEST_F(TestChannelsExtractorLUV, TestNaturalRGBImageOpenCL)
//{
//  cv::Mat img = cv::imread("images/index.jpeg");

//  compareLUVWithPDollar(img, channExtractorPDollar, channExtractorOpenCL);
//  compareLUVWithPDollar(img, channExtractorSmoothPDollar, channExtractorSmoothOpenCL);
//}

//TEST_F(TestChannelsExtractorLUV, TestNaturalSmoothRGBImageOpenCL)
//{
//  cv::Mat img = cv::imread("images/index.jpeg");

//  compareLUVWithPDollar(img, channExtractorPDollar, channExtractorOpenCL);
//  compareLUVWithPDollar(img, channExtractorSmoothPDollar, channExtractorSmoothOpenCL);
//}


// -------------------------------------------------------------------------------
// Test LUV PDollar.
// -------------------------------------------------------------------------------
TEST_F(TestChannelsExtractorLUV, TestWhiteImagePDollar)
{
  // Setting white image (255,255, 255) in all pixels .
  cv::Mat img_white(cv::Size(COLS, ROWS), CV_8UC3, cv::Scalar(255, 255, 255));

  cv::Mat img_white_luv_gt = cv::Mat::zeros(cv::Size(COLS, ROWS), CV_32FC3);

  //std::cout << "img_white=" << img_white << std::endl;

  for (int i=0; i<ROWS; i++)
  {
    for (int j=0; j<COLS; j++)
    {
      // Ground truth from Matlab Toolbox (Piotr Dollar):
      //   I = zeros(2,4,3);
      //   luv = rgbConvert(I, 'luv');
      img_white_luv_gt.at<cv::Vec3f>(i,j) = cv::Vec3f(0.3703704f, 0.3259259f, 0.4962940f);
    }
  }
  std::vector<cv::Mat> channels = channExtractorPDollar->extractFeatures(img_white);

//#ifdef VISUALIZE_RESULTS
//  cv::imshow("L", channels[0]);
//  cv::imshow("U", channels[1]);
//  cv::imshow("V", channels[2]);
//  cv::waitKey();
//#endif

  for (int i=0; i<ROWS; i++)
  {
    for (int j=0; j<COLS; j++)
    {
      cv::Vec3f gt = img_white_luv_gt.at<cv::Vec3f>(i,j);
      cv::Vec3f estimated(channels[0].at<float>(i,j), channels[1].at<float>(i,j), channels[2].at<float>(i,j));

      ASSERT_TRUE(abs(gt[0] - estimated[0]) < 1.e-6f);
      ASSERT_TRUE(abs(gt[1] - estimated[1]) < 1.e-6f);
      ASSERT_TRUE(abs(gt[2] - estimated[2]) < 1.e-6f);
    }
  }
}

TEST_F(TestChannelsExtractorLUV, TestBlackImagePDollar)
{
  // Setting black image (0,0,0) in all pixels.
  cv::Mat img_black(cv::Size(COLS, ROWS), CV_8UC3, cv::Scalar(0,0,0));
  cv::Mat img_black_luv_gt = cv::Mat::zeros(cv::Size(COLS, ROWS), CV_32FC3);
  for (int i=0; i<ROWS; i++)
  {
    for (int j=0; j<COLS; j++)
    {
      // Ground truth from Matlab Toolbox (Piotr Dollar):
      //   I = ones(2,4,3); % in float 1 is max gray level.
      //   luv = rgbConvert(I, 'luv');
      img_black_luv_gt.at<cv::Vec3f>(i,j) = cv::Vec3f(0.0f, 0.3259259f, 0.4962963f);
    }
  }
  std::vector<cv::Mat> channels = channExtractorPDollar->extractFeatures(img_black);

//#ifdef VISUALIZE_RESULTS
//  cv::imshow("L", channels[0]);
//  cv::imshow("U", channels[1]);
//  cv::imshow("V", channels[2]);
//  cv::waitKey();
//#endif

  for (int i=0; i<ROWS; i++)
  {
    for (int j=0; j<COLS; j++)
    {
      cv::Vec3f gt = img_black_luv_gt.at<cv::Vec3f>(i,j);
      cv::Vec3f estimated(channels[0].at<float>(i,j), channels[1].at<float>(i,j), channels[2].at<float>(i,j));

      ASSERT_TRUE(abs(gt[0] - estimated[0]) < 1.e-6f);
      ASSERT_TRUE(abs(gt[1] - estimated[1]) < 1.e-6f);
      ASSERT_TRUE(abs(gt[2] - estimated[2]) < 1.e-6f);
    }
  }
}

TEST_F(TestChannelsExtractorLUV, TestMediumGrayImagePDollar)
{
  // Setting medium gray image (128, 128, 128) in all pixels .
  cv::Mat img_gray(cv::Size(COLS, ROWS), CV_8UC3, cv::Scalar(128, 128, 128));
  cv::Mat img_gray_luv_gt = cv::Mat::zeros(cv::Size(COLS, ROWS), CV_32FC3);
  for (int i=0; i<ROWS; i++)
  {
    for (int j=0; j<COLS; j++)
    {
      // Ground truth from Matlab Toolbox (Piotr Dollar):
      //   I = ones(2,4,3)*(128/255); % in float 1 is max gray level, (128/255) is medium.
      //   luv = rgbConvert(I, 'luv');
      img_gray_luv_gt.at<cv::Vec3f>(i,j) = cv::Vec3f(0.2821814f, 0.3259259f, 0.4962947f);
    }
  }

  std::vector<cv::Mat> channels = channExtractorPDollar->extractFeatures(img_gray);

//#ifdef VISUALIZE_RESULTS
//  cv::imshow("L", channels[0]);
//  cv::imshow("U", channels[1]);
//  cv::imshow("V", channels[2]);
//  cv::waitKey();
//#endif

  for (int i=0; i<ROWS; i++)
  {
    for (int j=0; j<COLS; j++)
    {
      cv::Vec3f gt = img_gray_luv_gt.at<cv::Vec3f>(i,j);
      cv::Vec3f estimated(channels[0].at<float>(i,j), channels[1].at<float>(i,j), channels[2].at<float>(i,j));

//      std::cout << "gt (" << i << "," << j << ") = " << gt << std::endl;
//      std::cout << "estimated (" << i << "," << j << ") = " << estimated << std::endl;

      ASSERT_TRUE(abs(gt[0] - estimated[0]) < 1.e-6f);
      ASSERT_TRUE(abs(gt[1] - estimated[1]) < 1.e-6f);
      ASSERT_TRUE(abs(gt[2] - estimated[2]) < 1.e-6f);
    }
  }
}

TEST_F(TestChannelsExtractorLUV, TestRedImagePDollar)
{
  cv::Mat img(cv::Size(COLS, ROWS), CV_8UC3, cv::Scalar(0, 0, 255)); // BRG
  cv::Mat img_luv_gt = cv::Mat::zeros(cv::Size(COLS, ROWS), CV_32FC3);
  for (int i=0; i<ROWS; i++)
  {
    for (int j=0; j<COLS; j++)
    {
      // Ground truth from Matlab Toolbox (Piotr Dollar):
      //   I = zeros(2,4,3); % in float 1 is max gray level
      //   I(:,:,1) = ones(2,4); % in float 1 is max gray level
      //   luv = rgbConvert(I, 'luv');
      img_luv_gt.at<cv::Vec3f>(i,j) = cv::Vec3f(0.2007584f, 0.9858830f, 0.6386809f);
    }
  }

  std::vector<cv::Mat> channels = channExtractorPDollar->extractFeatures(img);

//#ifdef VISUALIZE_RESULTS
//  cv::imshow("L", channels[0]);
//  cv::imshow("U", channels[1]);
//  cv::imshow("V", channels[2]);
//  cv::waitKey();
//#endif

  for (int i=0; i<ROWS; i++)
  {
    for (int j=0; j<COLS; j++)
    {
      cv::Vec3f gt = img_luv_gt.at<cv::Vec3f>(i,j);
      cv::Vec3f estimated(channels[0].at<float>(i,j), channels[1].at<float>(i,j), channels[2].at<float>(i,j));

//      std::cout << "gt (" << i << "," << j << ") = " << gt << std::endl;
//      std::cout << "estimated (" << i << "," << j << ") = " << estimated << std::endl;

      ASSERT_TRUE(abs(gt[0] - estimated[0]) < 1.e-6f);
      ASSERT_TRUE(abs(gt[1] - estimated[1]) < 1.e-6f);
      ASSERT_TRUE(abs(gt[2] - estimated[2]) < 1.e-6f);
    }
  }
}

TEST_F(TestChannelsExtractorLUV, TestBlueImagePDollar)
{
  cv::Mat img(cv::Size(COLS, ROWS), CV_8UC3, cv::Scalar(255, 0, 0)); // BRG
  cv::Mat img_luv_gt = cv::Mat::zeros(cv::Size(COLS, ROWS), CV_32FC3);
  for (int i=0; i<ROWS; i++)
  {
    for (int j=0; j<COLS; j++)
    {
      // Ground truth from Matlab Toolbox (Piotr Dollar):
      //   I = zeros(2,4,3); % in float 1 is max gray level
      //   I(:,:,1) = ones(2,4); % in float 1 is max gray level
      //   luv = rgbConvert(I, 'luv');
      img_luv_gt.at<cv::Vec3f>(i,j) = cv::Vec3f(0.1188836f, 0.2913153f, 0.0165206f);
    }
  }

  std::vector<cv::Mat> channels = channExtractorPDollar->extractFeatures(img);

//#ifdef VISUALIZE_RESULTS
//  cv::imshow("L", channels[0]);
//  cv::imshow("U", channels[1]);
//  cv::imshow("V", channels[2]);
//  cv::waitKey();
//#endif

  for (int i=0; i<ROWS; i++)
  {
    for (int j=0; j<COLS; j++)
    {
      cv::Vec3f gt = img_luv_gt.at<cv::Vec3f>(i,j);
      cv::Vec3f estimated(channels[0].at<float>(i,j), channels[1].at<float>(i,j), channels[2].at<float>(i,j));

//      std::cout << "gt (" << i << "," << j << ") = " << gt << std::endl;
//      std::cout << "estimated (" << i << "," << j << ") = " << estimated << std::endl;

      ASSERT_TRUE(abs(gt[0] - estimated[0]) < 1.e-6f);
      ASSERT_TRUE(abs(gt[1] - estimated[1]) < 1.e-6f);
      ASSERT_TRUE(abs(gt[2] - estimated[2]) < 1.e-6f);
    }
  }
}

TEST_F(TestChannelsExtractorLUV, TestGreenImagePDollar)
{
  cv::Mat img(cv::Size(COLS, ROWS), CV_8UC3, cv::Scalar(0, 255, 0)); // BRG
  cv::Mat img_luv_gt = cv::Mat::zeros(cv::Size(COLS, ROWS), CV_32FC3);
  for (int i=0; i<ROWS; i++)
  {
    for (int j=0; j<COLS; j++)
    {
      // Ground truth from Matlab Toolbox (Piotr Dollar):
      //   I = zeros(2,4,3); % in float 1 is max gray level
      //   I(:,:,1) = ones(2,4); % in float 1 is max gray level
      //   luv = rgbConvert(I, 'luv');
      img_luv_gt.at<cv::Vec3f>(i,j) = cv::Vec3f(0.3233073f, 0.0012410f, 0.8871732f);
    }
  }

  std::vector<cv::Mat> channels = channExtractorPDollar->extractFeatures(img);

//#ifdef VISUALIZE_RESULTS
//  cv::imshow("L", channels[0]);
//  cv::imshow("U", channels[1]);
//  cv::imshow("V", channels[2]);
//  cv::waitKey();
//#endif

  for (int i=0; i<ROWS; i++)
  {
    for (int j=0; j<COLS; j++)
    {
      cv::Vec3f gt = img_luv_gt.at<cv::Vec3f>(i,j);
      cv::Vec3f estimated(channels[0].at<float>(i,j), channels[1].at<float>(i,j), channels[2].at<float>(i,j));

//      std::cout << "gt (" << i << "," << j << ") = " << gt << std::endl;
//      std::cout << "estimated (" << i << "," << j << ") = " << estimated << std::endl;

      ASSERT_TRUE(abs(gt[0] - estimated[0]) < 1.e-6f);
      ASSERT_TRUE(abs(gt[1] - estimated[1]) < 1.e-6f);
      ASSERT_TRUE(abs(gt[2] - estimated[2]) < 1.e-6f);
    }
  }
}

TEST_F(TestChannelsExtractorLUV, TestNaturalRGBImagePDollar)
{
  cv::Mat img_natural;
  img_natural = cv::imread("images/index.jpeg");

  std::vector<cv::Mat> vec_luv_gt(3);

  cv::FileStorage fs("yaml/luv_matlab_index.jpeg.yml", cv::FileStorage::READ);
  fs["luv_1"] >> vec_luv_gt[0];
  fs["luv_2"] >> vec_luv_gt[1];
  fs["luv_3"] >> vec_luv_gt[2];

  cv::Mat img_luv_gt;
  cv::merge(vec_luv_gt, img_luv_gt);


  ASSERT_FALSE(img_natural.empty());
  std::vector<cv::Mat> channels = channExtractorPDollar->extractFeatures(img_natural);

//#ifdef VISUALIZE_RESULTS
//  cv::imshow("L,gt", vec_luv_gt[0]);
//  cv::imshow("U,gt", vec_luv_gt[1]);
//  cv::imshow("V,gt", vec_luv_gt[2]);
//  cv::waitKey();

//  cv::imshow("L", channels[0]);
//  cv::imshow("U", channels[1]);
//  cv::imshow("V", channels[2]);
//  cv::waitKey();
//#endif
  //printf("NumCols %d %d \n", img_natural.rows, img_natural.cols);

  for (int i=0; i<img_natural.rows; i++)
  {
    for (int j=0; j<img_natural.cols; j++)
    {
      cv::Vec3f gt = img_luv_gt.at<cv::Vec3f>(i,j);
      cv::Vec3f estimated(channels[0].at<float>(i,j), channels[1].at<float>(i,j), channels[2].at<float>(i,j));

      //std::cout << "gt (" << i << "," << j << ") = " << gt << std::endl;
      //std::cout << "estimated (" << i << "," << j << ") = " << estimated << std::endl;

      ASSERT_TRUE(abs(gt[0] - estimated[0]) < 1.e-6f);
      ASSERT_TRUE(abs(gt[1] - estimated[1]) < 1.e-6f);
      ASSERT_TRUE(abs(gt[2] - estimated[2]) < 1.e-6f);
    }
  }
}

TEST_F(TestChannelsExtractorLUV, TestNaturalSmoothRGBImagePDollar)
{
  cv::Mat img_natural;
  img_natural = cv::imread("images/index.jpeg");


  std::vector<cv::Mat> vec_luv_gt(3);

  cv::FileStorage fs("yaml/luv_matlab_index.jpeg.yml", cv::FileStorage::READ);
  fs["luv_1"] >> vec_luv_gt[0];
  fs["luv_2"] >> vec_luv_gt[1];
  fs["luv_3"] >> vec_luv_gt[2];

  cv::Mat img_luv_gt;
  cv::merge(vec_luv_gt, img_luv_gt);


  ASSERT_FALSE(img_natural.empty());
  std::vector<cv::Mat> channels = channExtractorSmoothPDollar->extractFeatures(img_natural);


//#ifdef VISUALIZE_RESULTS
//  cv::imshow("L,gt", vec_luv_gt[0]);
//  cv::imshow("U,gt", vec_luv_gt[1]);
//  cv::imshow("V,gt", vec_luv_gt[2]);
//  cv::waitKey();

//  cv::imshow("L", channels[0]);
//  cv::imshow("U", channels[1]);
//  cv::imshow("V", channels[2]);
//  cv::waitKey();
//#endif
  //printf("NumCols %d %d \n", img_natural.rows, img_natural.cols);
  //printf("%.4f\n", channels[0].at<float>(0,0));

  cv::FileStorage fs_smooth;

  fs.open("yaml/smoothImage_L.yaml", cv::FileStorage::READ);

  cv::FileNode rows = fs["smoothL"]["rows"];
  cv::FileNode smoothMatlab = fs["smoothL"]["data"];

  for(int i=0; i< (int)10; i++)
  {
    for(int j=0; j<(int)2; j++)
    {
      ASSERT_TRUE(abs(channels[0].at<float>(j,i) - static_cast<float>(smoothMatlab[i*static_cast<int>(rows)+j])) < 1.e-4f);
      //printf("%.4f %.4f\n", channels[0].at<float>(j,i), (float)smoothMatlab[i*(int)rows+j]);
    }
  }
}
