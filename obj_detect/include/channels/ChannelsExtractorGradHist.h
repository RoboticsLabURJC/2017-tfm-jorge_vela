
/** ------------------------------------------------------------------------
 *
 *  @brief Channel feature extractors for histogram gradients
 *  @author Jorge Vela
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2019/07/08
 *
 *  ------------------------------------------------------------------------ */

#ifndef CHANNELS_EXTRACTOR_GRAD_HIST
#define CHANNELS_EXTRACTOR_GRAD_HIST

#include <opencv2/opencv.hpp>
#include <vector>

class ChannelsExtractorGradHist
{

public:
  ChannelsExtractorGradHist
    (
      int binSize = 8,
      int nOrients = 8,
      int softBin = 1,
      int full = 0,
      bool use_opencv_impl = true
    ) 
  {
    m_binSize = binSize;
    m_nOrients = nOrients;
    m_softBin = softBin;
    m_full = full;
    m_use_opencv_impl = use_opencv_impl;
  };

  virtual ~ChannelsExtractorGradHist() {};

  virtual std::vector<cv::Mat> extractFeatures
    (
      cv::Mat img, 
      std::vector<cv::Mat> gradMag
    ) = 0;

protected:
	int m_binSize;
	int m_nOrients;
	int m_softBin;
	int m_full;
    bool m_use_opencv_impl;
};

#endif
