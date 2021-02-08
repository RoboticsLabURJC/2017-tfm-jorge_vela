/** ------------------------------------------------------------------------
 *
 *  @brief ChannelsPyramid.
 *  @author Jorge Vela
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2020/06/01
 *
 *  ------------------------------------------------------------------------ */

#ifndef CHANNELS_PYRAMID
#define CHANNELS_PYRAMID

#include <opencv2/opencv.hpp>
#include <detectors/ClassifierConfig.h>
#include <vector>
#include <string>
#include <iostream>

/** ------------------------------------------------------------------------
 *
 *  @brief Abstract class for LDCF or ACF pyramid extraction.
 *
 *  ------------------------------------------------------------------------ */
class ChannelsPyramid
{
public:
  ChannelsPyramid
    (
    std::string channels_impl_type
    ) { m_channels_impl_type = channels_impl_type; }

  virtual ~ChannelsPyramid
  () {};

  virtual std::vector<std::vector<cv::Mat>> compute
    (
    cv::Mat img,
    std::vector<cv::Mat> filters,
    std::vector<double>& scales,
    std::vector<cv::Size2d>& scaleshw,
    ClassifierConfig clf
    ) = 0;

  virtual int getScales
    (
    int nPerOct,
    int nOctUp,
    const cv::Size& minDs,
    int shrink,
    const cv::Size& sz,
    std::vector<double>& scales,
    std::vector<cv::Size2d>& scaleshw
    );

  /**
   * @brief createChannelsPyramid Factory class method for channel pyramid computation algorithms
   *
   * The implementation strings for different strategies are:
   *   - "all" for ChannelsPyramidComputeAllStrategy
   *   - "all_parallel" for ChannelsPyramidComputeAllParallelStrategy
   *   - "approximated" for ChannelsPyramidApproximatedtrategy
   *   - "approximated_parallel" for ChannelsPyramidApproximatedParallelStrategy
   *
   * @param pyramid_impl_type std::string with the name of the strategy.
   * @return shared_ptr to a ChannelsPyramid implementation.
   */
  static std::shared_ptr<ChannelsPyramid>
  createChannelsPyramid
    (
    std::string pyramid_impl_type = "approximated_parallel",
    std::string channels_impl_type = "pdollar"
    );

protected:
  std::string m_channels_impl_type;
//  int m_nOctUp;
//  int m_nPerOct;
//  int m_nApprox;
//  int m_shrink;

//  int m_gradientMag_normRad;
//  float m_gradientMag_normConst;

//  int m_gradientHist_binSize;
//  int m_gradientHist_nOrients;
//  int m_gradientHist_softBin;
//  int m_gradientHist_full;

//  std::vector<float> m_lambdas;
//  cv::Size m_padding;
//  cv::Size m_minDs;
  
};


#endif
