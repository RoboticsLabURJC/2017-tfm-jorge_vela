/** ------------------------------------------------------------------------
 *
 *  @brief Channel feature extractors for magnitude and orient gradients.
 *  @author Jorge Vela
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2019/07/08
 *
 *  ------------------------------------------------------------------------ */

#ifndef CHANNELS_EXTRACTOR_GRAD_MAG_PDOLLAR
#define CHANNELS_EXTRACTOR_GRAD_MAG_PDOLLAR

#include <channels/ChannelsExtractorGradMag.h>
#include <opencv2/opencv.hpp>
#include <vector>

class ChannelsExtractorGradMagPDollar: public ChannelsExtractorGradMag
{
public:
  ChannelsExtractorGradMagPDollar
    (
      int normRad = 0,
      float normConst = 0.005
    ): ChannelsExtractorGradMag(normRad, normConst)
  {};
    
  virtual ~ChannelsExtractorGradMagPDollar() {};

  virtual std::vector<cv::Mat> extractFeatures
    (
      cv::Mat img 
    );
};

#endif
