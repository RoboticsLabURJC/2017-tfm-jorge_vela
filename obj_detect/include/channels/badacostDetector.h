/** ------------------------------------------------------------------------
 *
 *  @brief badacostDetector.
 *  @author Jorge Vela
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2020/06/01
 *
 *  ------------------------------------------------------------------------ */


#ifndef BADACOST_DETECTOR
#define BADACOST_DETECTOR

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
//#include "gtest/gtest.h"
#include <iostream>
#include <channels/ChannelsPyramid.h>

class BadacostDetector
{
public:
  BadacostDetector
    (
    int shrink = 2, 
    int modelHt = 2, 
    int modelWd = 2, 
    int stride = 2, 
    int cascThr = 2
    )
    {   
      m_srhink = shrink;
      m_modelHt = modelHt;
      m_modelWd = modelWd;
      m_stride = stride;
      m_cascThr = cascThr;
      m_classifierIsLoaded = false;
    };

  // Loads the trained classfier from a yaml file.
  bool load(std::string clfPath, std::string pyrPath, std::string filtersPath);

  // Detects objects at any scale in an image and return detections as rectangles
  std::vector<cv::Rect2i> detect(cv::Mat img);

protected:
  std::vector<cv::Mat> loadFilters(std::string filtersPath);

  // Detects objects at a single scale in an image and return detections as rectangles
//  std::vector<cv::Rect2i> detectSingleScale(cv::Mat img);

  bool m_classifierIsLoaded;
    
  int m_srhink;
  int m_modelHt;
  int m_modelWd;
  int m_stride;
  int m_cascThr;

  // From here, this variables are read from a yaml file obtained from 
  // trained detector in Matlab.
  int m_treeDepth;
  int m_num_classes;
  int m_aRatioFixedWidth;    
    
  cv::Mat m_Cprime;
  cv::Mat m_Y;
  cv::Mat m_wl_weights;
  cv::Mat m_aRatio;
  

  // After load it contains the cv::Mat variables: 
  // "fids", "thrs", "child", "hs", "weights", "depth"
  std::map<std::string, cv::Mat> m_classifier;

  // After load it contains the variables for build the pyramid of channel features
  // "fids", "thrs", "child", "hs", "weights", "depth"
  ChannelsPyramid m_chnsPyramid;

  std::vector<cv::Mat> m_filters;
};


#endif

