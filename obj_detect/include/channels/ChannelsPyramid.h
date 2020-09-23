#ifndef CHANNELS_PYRAMID
#define CHANNELS_PYRAMID

#include <opencv2/opencv.hpp>

#include <vector>
#include <string>

#include "gtest/gtest.h"
#include <opencv2/opencv.hpp>

#include <iostream>

class ChannelsPyramid
{
public:
  ChannelsPyramid() {}
  bool load(std::string opts);

  std::vector<cv::Mat> getPyramid(cv::Mat img);
  std::vector<cv::Mat> badacostFilters(cv::Mat pyramid, std::vector<cv::Mat> filters);
//  std::vector<float> getScales(int nPerOct, int nOctUp, int minDs[], int shrink, int sz[]);

  int getScales(
  int nPerOct,
  int nOctUp,
  const cv::Size& minDs,
  int shrink,
  const cv::Size& sz,
  std::vector<double>& scales,
  std::vector<cv::Size2d>& scaleshw
  );


private:

  int m_nOctUp;
  int m_nPerOct;
  int m_nApprox;
  int m_shrink;
  std::vector<int> m_minDs;
};


#endif
