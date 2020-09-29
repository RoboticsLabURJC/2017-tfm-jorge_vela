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
    ();

  virtual ~ChannelsPyramid
    ();

  virtual bool load
    (
    std::string opts
    );

  virtual std::vector<std::vector<cv::Mat>> compute
    (
    cv::Mat img,
    std::vector<cv::Mat> filters
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

protected:
  int m_nOctUp;
  int m_nPerOct;
  int m_nApprox;
  int m_shrink;
  std::vector<int> m_minDs;
};


#endif
