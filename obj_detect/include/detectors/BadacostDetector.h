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

#include <detectors/DetectionRectangle.h>
#include <pyramid/ChannelsPyramid.h>
#include <pyramid/ChannelsPyramidComputeAllStrategy.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>

class BadacostDetector
{
public:
  BadacostDetector
    (
    ChannelsPyramid* pChnsPyramidStrategy = nullptr
    );

  ~BadacostDetector
    ();

  // Loads the trained classfier from a yaml file.
  bool load
    (
    std::string clfPath,
    std::string pyrPath,
    std::string filtersPath
    );

  // Detects objects at any scale in an image and return detections as rectangles
  std::vector<DetectionRectangle>
  detect
    (
    cv::Mat img
    );

protected:
  std::vector<cv::Mat> loadFilters(std::string filtersPath);

  // Detects objects at a single scale in an image and return detections as rectangles
  std::vector<DetectionRectangle>
  detectSingleScale
    (
    std::vector<cv::Mat>& channels
    );

  void correctToClassSpecificBbs
    (
    std::vector<DetectionRectangle>& dts,
    std::vector<float> aRatios,
    bool fixedWidth = false // In this case we keep the h fixed and modify w
    );

  bool m_classifierIsLoaded;
    
  int m_shrink;
  int m_stride;
  float m_cascThr;
  cv::Size m_modelDsPad;
  cv::Size m_modelDs;
  cv::Size m_padding;

  // From here, this variables are read from a yaml file obtained from 
  // trained detector in Matlab.
  int m_treeDepth;
  int m_num_classes;
  bool m_aRatioFixedWidth;
    
  cv::Mat m_Cprime;
  cv::Mat m_Y;
  cv::Mat m_wl_weights;
  cv::Mat m_aRatio;
  
  // After load it contains the cv::Mat variables: 
  // "fids", "thrs", "child", "hs", "weights", "depth"
  std::map<std::string, cv::Mat> m_classifier;

  // After load it contains the variables for build the pyramid of channel features
  // "fids", "thrs", "child", "hs", "weights", "depth"
  ChannelsPyramid* m_pChnsPyramidStrategy;

  std::vector<cv::Mat> m_filters;
};


#endif

