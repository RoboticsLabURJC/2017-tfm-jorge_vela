#ifndef CHANNELS_PYRAMID
#define CHANNELS_PYRAMID

#include <opencv/cv.hpp>
#include <vector>
#include <string>

#include "gtest/gtest.h"
#include <opencv/cv.hpp>

#include <iostream>

class ChannelsPyramid
{
private:
  int m_nOctUp;
  int m_nPerOct;
  int m_nApprox;
  int m_shrink;
  std::vector<int> m_minDs;



protected:



public:
	ChannelsPyramid(
	){
	}
    std::vector<float> getScales(  int nPerOct, int nOctUp, int minDs[], int shrink, int sz[]);

	bool load(std::string opts);

	std::vector<cv::Mat> getPyramid(cv::Mat img, int nOctUp, int nPerOct, int nApprox, int shrink);

};


#endif
