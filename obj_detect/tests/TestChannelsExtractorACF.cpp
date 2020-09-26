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
  cv::Mat image = cv::imread("images/coche_solo1.png", cv::IMREAD_COLOR);

  // Prepare reading the yaml file with Matlab's results.
  FileStorage fs1;
  bool file_exists = fs1.open("yaml/acfChannelsScale0_coche_solo1_png.yaml", FileStorage::READ);
  ASSERT_TRUE(file_exists);

  // Read channel shrink used in matlab:
  cv::FileNode shrinkFileNode = fs1["shrink"]["data"];
  std::vector<float> p;
  shrinkFileNode >> p;
  int shrink = p[0];
  std::cout << "shrink=" << shrink << std::endl;

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

  std::cout << "padding=" << padding << std::endl;

  // Extract ACF channels using paramenters from matlab.
  std::vector<cv::Mat> acf_channels;
  ChannelsExtractorACF acfExtractor(padding, shrink, "RGB");
  acf_channels = acfExtractor.extractFeatures(image);

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

#ifdef DEBUG
    std::cout << "rows=" << rows << std::endl;
    std::cout << "cols=" << cols << std::endl;
    std::cout << "channel_matlab.size()" << channel_matlab.size() << std::endl;
    std::cout << "acf_channels[3].size()" << acf_channels[3].size() << std::endl;
#endif
    cv::Mat absDiff = cv::abs(channel_matlab - acf_channels[i]);

    // Find the max absolute difference between the elements in the ACF channels w.r.t.
    // Matlab's values:
    double min_abs_diff;
    double max_abs_diff;
    int min_ind[2];
    int max_ind[2];
    cv::minMaxIdx(absDiff, &min_abs_diff, &max_abs_diff, min_ind, max_ind, cv::Mat());
    ASSERT_TRUE(max_abs_diff < 1.e-2f);
  }
}
