
/** ------------------------------------------------------------------------
 *
 *  @brief Channel feature extractors for histogram gradients
 *  @author Jorge Vela
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2019/07/08
 *
 *  ------------------------------------------------------------------------ */

#ifndef CHANNELS_GRADHIST_OCL
#define CHANNELS_GRADHIST_OCL

#include <channels/ChannelsExtractorGradHist.h>
#include <opencv2/opencv.hpp>
#include <vector>


class ChannelsExtractorGradHistOpenCL: public ChannelsExtractorGradHist
{

public:
  ChannelsExtractorGradHistOpenCL
    (
      int binSize = 8,
      int nOrients = 8,
      int softBin = 1,
      int full = 0
    ): ChannelsExtractorGradHist(binSize,
                               nOrients,
                               softBin,
                               full = 0)
    {};

  virtual std::vector<cv::Mat> extractFeatures
    (
      cv::Mat img,
      std::vector<cv::Mat> gradMag
    );

  std::vector<cv::UMat> extractFeatures
    (
      cv::UMat img,
      std::vector<cv::UMat> gradMag
    );


private:

  void
  gradQuantize
    (
      cv::UMat O,
      cv::UMat M,
      float norm,
      int nOrients,
      bool full,
      bool interpolate,
      cv::UMat& O0,
      cv::UMat& O1,
      cv::UMat& M0,
      cv::UMat& M1
    );

  void
  gradHist
    (
      cv::UMat M,
      cv::UMat O,
      std::vector<cv::UMat>& H,
      int bin,
      int nOrients,
      int softBin,
      bool full
    );
};

#endif
