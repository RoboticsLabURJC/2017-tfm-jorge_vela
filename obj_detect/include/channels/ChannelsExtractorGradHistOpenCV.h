
/** ------------------------------------------------------------------------
 *
 *  @brief Channel feature extractors for histogram gradients
 *  @author Jorge Vela
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2019/07/08
 *
 *  ------------------------------------------------------------------------ */

#ifndef CHANNELS_GRADHIST
#define CHANNELS_GRADHIST

#include <channels/ChannelsExtractorGradHist.h>
#include <opencv2/opencv.hpp>
#include <vector>

class ChannelsExtractorGradHistOpenCV: public ChannelsExtractorGradHist
{

public:
  ChannelsExtractorGradHistOpenCV
    (
      int binSize = 8,
      int nOrients = 8,
      int softBin = 1,
      int full = 0
    ); /*: ChannelsExtractorGradHist(binSize,
                               nOrients,
                               softBin,
                               full = 0)
    {};*/

  virtual std::vector<cv::Mat> extractFeatures
    (
      cv::Mat img,
      std::vector<cv::Mat> gradMag
    );

private:
  cv::Mat m_kernel;
  cv::Mat m_kernel1d;

  void
  createKernel
    (
    int bin,
    int nOrients,
    int softBin,
    bool full
    );

  void
  gradQuantize
    (
      cv::Mat O,
      cv::Mat M,
      float norm,
      int nOrients,
      bool full,
      bool interpolate,
      cv::Mat& O0,
      cv::Mat& O1,
      cv::Mat& M0,
      cv::Mat& M1
    );

  void
  gradHist
    (
      cv::Mat M,
      cv::Mat O,
      std::vector<cv::Mat>& H,
      int bin,
      int nOrients,
      int softBin,
      bool full
    );
};

#endif
