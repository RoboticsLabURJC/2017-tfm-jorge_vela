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

#include <detectors/ClassifierConfig.h>
#include <detectors/DetectionRectangle.h>
#include <pyramid/ChannelsPyramid.h>
#include <pyramid/ChannelsPyramidOpenCL.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>

class BadacostDetector
{
public:
  BadacostDetector
    (
    std::string channels_pyramid_impl = "all_parallel",
    std::string channels_impl = "pdollar",
    float minScore = 0
    );

  ~BadacostDetector
    ();

  // Loads the trained classfier from a yaml file.
  bool load
    (
    std::string clfPath,
    //std::string pyrPath,
    std::string filtersPath
    );

  // Detects objects at any scale in an image and return detections as rectangles
  std::vector<DetectionRectangle>
  detect
    (
    cv::Mat img
    );

  void
  showResults
    (
    cv::Mat img,
    const std::vector<DetectionRectangle>& detections
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
  float m_minScore;  

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

  std::string m_channels_pyramid_impl;
  std::string m_channels_impl;

  // After load it contains the cv::Mat variables: 
  // "fids", "thrs", "child", "hs", "weights", "depth"
  std::map<std::string, cv::Mat> m_classifier;

  // After load it contains the variables for build the pyramid of channel features
  // "fids", "thrs", "child", "hs", "weights", "depth"
  std::shared_ptr<ChannelsPyramid> m_pChnsPyramidStrategy;
  ClassifierConfig m_clfData;

  // We use a completely different implementation of ChannelsPyramid for T-API (OpenCL)
  ChannelsPyramidOpenCL m_chnsPyramidOpenCL;

  std::vector<cv::Mat> m_filters;
};


#endif

