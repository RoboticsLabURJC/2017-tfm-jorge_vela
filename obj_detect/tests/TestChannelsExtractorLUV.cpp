/** ------------------------------------------------------------------------
 *
 *  @brief Test of Channels Extractor for LUV color space
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2019/07/08
 *
 *  ------------------------------------------------------------------------ */

#include <channels/ChannelsExtractorLUV.h>
#include "gtest/gtest.h"
#include <opencv/cv.hpp>

#undef VISUALIZE_RESULTS

class TestChannelsExtractorLUV: public testing::Test
{
 public:

  const int ROWS = 2;
  const int COLS = 4;
  ChannelsLUVExtractor channExtract{false, 1};

  virtual void SetUp()
  {
  }

  virtual void TearDown()
  {
  }
};

TEST_F(TestChannelsExtractorLUV, TestWhiteImage)
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
  std::vector<cv::Mat> channels = channExtract.extractFeatures(img_white);

#ifdef VISUALIZE_RESULTS
  cv::imshow("L", channels[0]);
  cv::imshow("U", channels[1]);
  cv::imshow("V", channels[2]);
  cv::waitKey();
#endif

  for (int i=0; i<ROWS; i++)
  {
    for (int j=0; j<COLS; j++)
    {
      cv::Vec3f gt = img_white_luv_gt.at<cv::Vec3f>(i,j);
      cv::Vec3f estimated(channels[0].at<float>(i,j), channels[1].at<float>(i,j), channels[2].at<float>(i,j));

//      std::cout << "gt (" << i << "," << j << ") = " << gt << std::endl;
//      std::cout << "estimated (" << i << "," << j << ") = " << estimated << std::endl;

      ASSERT_TRUE(abs(gt[0] - estimated[0]) < 1.e-6f);
      ASSERT_TRUE(abs(gt[1] - estimated[1]) < 1.e-6f);
      ASSERT_TRUE(abs(gt[2] - estimated[2]) < 1.e-6f);
    }
  }
}

TEST_F(TestChannelsExtractorLUV, TestBlackImage)
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
  std::vector<cv::Mat> channels = channExtract.extractFeatures(img_black);

#ifdef VISUALIZE_RESULTS
  cv::imshow("L", channels[0]);
  cv::imshow("U", channels[1]);
  cv::imshow("V", channels[2]);
  cv::waitKey();
#endif

  for (int i=0; i<ROWS; i++)
  {
    for (int j=0; j<COLS; j++)
    {
      cv::Vec3f gt = img_black_luv_gt.at<cv::Vec3f>(i,j);
      cv::Vec3f estimated(channels[0].at<float>(i,j), channels[1].at<float>(i,j), channels[2].at<float>(i,j));

//      std::cout << "gt (" << i << "," << j << ") = " << gt << std::endl;
//      std::cout << "estimated (" << i << "," << j << ") = " << estimated << std::endl;

      ASSERT_TRUE(abs(gt[0] - estimated[0]) < 1.e-6f);
      ASSERT_TRUE(abs(gt[1] - estimated[1]) < 1.e-6f);
      ASSERT_TRUE(abs(gt[2] - estimated[2]) < 1.e-6f);
    }
  }
}

TEST_F(TestChannelsExtractorLUV, TestMediumGrayImage)
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

  std::vector<cv::Mat> channels = channExtract.extractFeatures(img_gray);

#ifdef VISUALIZE_RESULTS
  cv::imshow("L", channels[0]);
  cv::imshow("U", channels[1]);
  cv::imshow("V", channels[2]);
  cv::waitKey();
#endif

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

TEST_F(TestChannelsExtractorLUV, TestRedImage)
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

  std::vector<cv::Mat> channels = channExtract.extractFeatures(img);

#ifdef VISUALIZE_RESULTS
  cv::imshow("L", channels[0]);
  cv::imshow("U", channels[1]);
  cv::imshow("V", channels[2]);
  cv::waitKey();
#endif

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

TEST_F(TestChannelsExtractorLUV, TestBlueImage)
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

  std::vector<cv::Mat> channels = channExtract.extractFeatures(img);

#ifdef VISUALIZE_RESULTS
  cv::imshow("L", channels[0]);
  cv::imshow("U", channels[1]);
  cv::imshow("V", channels[2]);
  cv::waitKey();
#endif

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

TEST_F(TestChannelsExtractorLUV, TestGreenImage)
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

  std::vector<cv::Mat> channels = channExtract.extractFeatures(img);

#ifdef VISUALIZE_RESULTS
  cv::imshow("L", channels[0]);
  cv::imshow("U", channels[1]);
  cv::imshow("V", channels[2]);
  cv::waitKey();
#endif

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

TEST_F(TestChannelsExtractorLUV, TestNaturalRGBImage)
{  
  cv::Mat img_natural;
  img_natural = cv::imread("index.jpeg");

  std::vector<cv::Mat> vec_luv_gt(3);

  cv::FileStorage fs("luv_matlab_index.jpeg.yml", cv::FileStorage::READ);
  fs["luv_1"] >> vec_luv_gt[0];
  fs["luv_2"] >> vec_luv_gt[1];
  fs["luv_3"] >> vec_luv_gt[2];

  cv::Mat img_luv_gt;
  cv::merge(vec_luv_gt, img_luv_gt);


  ASSERT_FALSE(img_natural.empty());
  std::vector<cv::Mat> channels = channExtract.extractFeatures(img_natural);

#ifdef VISUALIZE_RESULTS
  cv::imshow("L,gt", vec_luv_gt[0]);
  cv::imshow("U,gt", vec_luv_gt[1]);
  cv::imshow("V,gt", vec_luv_gt[2]);
  cv::waitKey();

  cv::imshow("L", channels[0]);
  cv::imshow("U", channels[1]);
  cv::imshow("V", channels[2]);
  cv::waitKey();
#endif
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


TEST_F(TestChannelsExtractorLUV, TestNaturalSmoothRGBImage)
{  
  cv::Mat img_natural;
  img_natural = cv::imread("index2.jpeg");

  std::vector<cv::Mat> vec_luv_gt(3);

  cv::FileStorage fs("luv_matlab_index.jpeg.yml", cv::FileStorage::READ);
  fs["luv_1"] >> vec_luv_gt[0];
  fs["luv_2"] >> vec_luv_gt[1];
  fs["luv_3"] >> vec_luv_gt[2];

  cv::Mat img_luv_gt;
  cv::merge(vec_luv_gt, img_luv_gt);


  ASSERT_FALSE(img_natural.empty());
  std::vector<cv::Mat> channels = channExtract.extractFeatures(img_natural);

#ifdef VISUALIZE_RESULTS
  cv::imshow("L,gt", vec_luv_gt[0]);
  cv::imshow("U,gt", vec_luv_gt[1]);
  cv::imshow("V,gt", vec_luv_gt[2]);
  cv::waitKey();

  cv::imshow("L", channels[0]);
  cv::imshow("U", channels[1]);
  cv::imshow("V", channels[2]);
  cv::waitKey();
#endif
  //printf("NumCols %d %d \n", img_natural.rows, img_natural.cols);
  
  for (int i=0; i<img_natural.rows; i++)
  {
    for (int j=0; j<img_natural.cols; j++)
    {
      cv::Vec3f gt = img_luv_gt.at<cv::Vec3f>(i,j);
      //printf("%f %f %f \n", channels[0].at<float>(i,j), channels[1].at<float>(i,j), channels[2].at<float>(i,j));
      cv::Vec3f estimated(channels[0].at<float>(i,j), channels[1].at<float>(i,j), channels[2].at<float>(i,j));

      //std::cout << "gt (" << i << "," << j << ") = " << gt << std::endl;
      //std::cout << "estimated (" << i << "," << j << ") = " << estimated << std::endl;

      //ASSERT_TRUE(abs(gt[0] - estimated[0]) < 1.e-6f);
      //ASSERT_TRUE(abs(gt[1] - estimated[1]) < 1.e-6f);
      //ASSERT_TRUE(abs(gt[2] - estimated[2]) < 1.e-6f);
    }
  }
}
