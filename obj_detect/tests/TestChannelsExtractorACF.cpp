/** ------------------------------------------------------------------------
 *
 *  @brief Test of Extraction of ACF Channels.
 *  @author Jorge Vela Peña
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2020/09/25
 *
 *  ------------------------------------------------------------------------ */
#include <channels/ChannelsExtractorACF.h>
#include <channels/ChannelsExtractorLUV.h>

#include "gtest/gtest.h"
#include <opencv2/opencv.hpp>
#include <iostream>

#undef DEBUG
//#define DEBUG

#undef SHOW_CHANNELS
//#define SHOW_CHANNELS

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
  cv::Mat image = cv::imread("images/coche_solo1.png", cv::IMREAD_COLOR);

#ifdef SHOW_CHANNELS
    cv::imshow("image", image);
    cv::waitKey();
#endif

  // We first pass the image to LUV:
//  ChannelsLUVExtractor luvExtractor;
//  std::vector<cv::Mat> luv_channels = luvExtractor.extractFeatures(image);
//  cv::Mat luv_image;
//  merge(luv_channels, luv_image);

//#ifdef SHOW_CHANNELS
//    cv::imshow("luv_image", luv_image);
//    cv::waitKey();
//#endif

  // Prepare reading the yaml file with Matlab's results.
  FileStorage fs1;
  bool file_exists = fs1.open("yaml/acfChannelsScale0_coche_solo1_png.yaml", FileStorage::READ);
  ASSERT_TRUE(file_exists);

  // Read channel shrink used in matlab:
  cv::FileNode shrinkFileNode = fs1["shrink"]["data"];
  std::vector<float> p;
  shrinkFileNode >> p;
  int shrink = p[0];

  // Read channel padding used in matlab:
  cv::Size padding;
  cv::FileNode padFileNode = fs1["pad_width"]["data"];
  p.clear();
  padFileNode >> p;
  padding.width = p[0];

  padFileNode = fs1["pad_height"]["data"];
  p.clear();
  padFileNode >> p;
  padding.height = p[0];

  // Extract ACF channels using paramenters from matlab.
  std::vector<cv::Mat> acf_channels;
  ChannelsExtractorACF acfExtractor(padding, shrink);
//  acf_channels = acfExtractor.extractFeatures(luv_image);
  acf_channels = acfExtractor.extractFeatures(image);

  cv::Size acf_channel_sz = acf_channels[0].size();
//  float crop_perct = 0.2;
//  cv::Size crop_sz(round(acf_channel_sz.width * crop_perct), round(acf_channel_sz.height * crop_perct));
//  cv::Rect crop_rect(crop_sz.width, crop_sz.height,
//                     acf_channel_sz.width - crop_sz.width*2, acf_channel_sz.height - crop_sz.height*2);
  for (int i=0; i < 10; i++) // read and compare all the channels
  {
    std::string var_name = "acf_channel_" + std::to_string(i+1);
    int rows = fs1[var_name]["rows"];
    int cols = fs1[var_name]["cols"];
    FileNode data = fs1[var_name]["data"];
    cv::Mat channel_matlab = cv::Mat::zeros(rows, cols, CV_32F);
    p.clear();
    data >> p;
    memcpy(channel_matlab.data, p.data(), p.size()*sizeof(float));

    cv::Mat absDiff = cv::abs(channel_matlab - acf_channels[i]);

#ifdef DEBUG
    std::cout << "rows=" << rows << std::endl;
    std::cout << "cols=" << cols << std::endl;
    std::cout << "channel_matlab.size()" << channel_matlab.size() << std::endl;
    std::cout << "acf_channels[3].size()" << acf_channels[3].size() << std::endl;
    std::cout << "channel_matlab =" << channel_matlab << std::endl;
    std::cout << "acf_channels[i] =" << acf_channels[i] << std::endl;
    std::cout << "absDiff = " << absDiff << std::endl;
#endif

#ifdef SHOW_CHANNELS
    cv::imshow("Matlab", channel_matlab);
    cv::imshow("CPP", acf_channels[i]);
    cv::imshow("absDiff", absDiff);
    cv::waitKey();
#endif

    double min_val;
    double max_val;
    int min_ind[2];
    int max_ind[2];
    cv::minMaxIdx(acf_channels[i], &min_val, &max_val, min_ind, max_ind, cv::Mat());
    float channel_range = max_val - min_val;
    float threshold = 0.2*channel_range; // Asume a 20% of the maximum value is an acceptable error.

    int num_differences = 0;
    for (int i = 0; i < acf_channel_sz.height; i++)
    {
      for (int j = 0; j < acf_channel_sz.width; j++)
      {
        if (absDiff.at<float>(i,j) > threshold)
        {
          num_differences++;
        }
      }
    }

  int num_pixels = acf_channel_sz.height*acf_channel_sz.height;
#ifdef DEBUG
  std::cout << "threshold=" << threshold << std::endl;
  std::cout << "num_pixels=" << num_pixels << std::endl;
  std::cout << "num_differences=" << num_differences << std::endl;
  std::cout << "num_pixels*0.3 =" << num_pixels*0.3 << std::endl;
#endif
  // Assume 30% of the pixels beyond the error threshold is acceptable.
  ASSERT_TRUE(num_differences < num_pixels*0.3);
  }
}
