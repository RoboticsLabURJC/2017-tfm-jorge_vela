
/** ------------------------------------------------------------------------
 *
 *  @brief Channel feature extractors for histogram gradients
 *  @author Jorge Vela
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2019/07/08
 *
 *  ------------------------------------------------------------------------ */

#ifndef CHANNELS_EXTRACTOR_GRAD_HIST_PDOLLAR
#define CHANNELS_EXTRACTOR_GRAD_HIST_PDOLLAR

#include <channels/ChannelsExtractorGradHist.h>
#include <opencv2/opencv.hpp>
#include <vector>


class ChannelsExtractorGradHistPDollar: public ChannelsExtractorGradHist
{

public:
  ChannelsExtractorGradHistPDollar
    (
      int binSize = 8,
      int nOrients = 8,
      int softBin = 1,
      int full = 0
    ): ChannelsExtractorGradHist(binSize,
                                 nOrients,
                                 softBin,
                                 full = 0)
  {
  };

  virtual std::vector<cv::Mat> extractFeatures
    (
      cv::Mat img, 
      std::vector<cv::Mat> gradMag
    );

private:
  std::vector<cv::Mat>
  gradH
    (
    cv::Mat image,
    float *M,
    float *O
    );

  void
  gradHist
    (
    float *M,
    float *O,
    float *H,
    int h,
    int w,
    int bin,
    int nOrients,
    int softBin,
    bool full
    );

  void
  gradQuantize
    (
    float *O,
    float *M,
    int *O0,
    int *O1,
    float *M0,
    float *M1,
    int nb,
    int n,
    float norm,
    int nOrients,
    bool full,
    bool interpolate
    );
};

#endif
