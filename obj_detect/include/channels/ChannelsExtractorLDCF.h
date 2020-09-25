/** ------------------------------------------------------------------------
 *
 *  @brief Locally Decorrelated Channel Features (LDCF) Extractor
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2020/09/25
 *
 *  ------------------------------------------------------------------------ */

#ifndef CHANNELS_EXTRACTOR_LDCF
#define CHANNELS_EXTRACTOR_LDCF

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

/** ------------------------------------------------------------------------
 *
 *  @brief Class for LDCF extraction: Incorrelated ACF (LUV, Gradient and HOG) channels.
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *
 *  ------------------------------------------------------------------------ */
class ChannelsExtractorLDCF
{
public:
   /**
    * This constructor sets the parameters for computing the ACF features and
    * the ones needed for LDCF.
    *
    * @param filters LDCF filters to be applied over the ACF channels.
    * @param padding for the ACF channels before filtering for LDCF
    * @param shrink
    */
  ChannelsExtractorLDCF
    (
      std::vector<cv::Mat> filters,
      cv::Size padding,
      int shrink
    );

  /**
   * This method computes all the Piotr Dollar's Locally Decorrelated Channel Features
   * from the Aggregated Channels Features as cv::Mat from an input image:
   *   - 3 color chanels in the LUV color space
   *   - 1 Gradient Magnitude channel
   *   - 6 HOG channels (6 orientations).
   * 
   * @param src input image
   * @return std::vector<cv::Mat> vector with all the channels as cv:Mat.
   */    
  std::vector<cv::Mat> extractFeatures
    (
      cv::Mat img // Should be a LUV image!!
    );

private:
  int m_shrink;
  std::string m_color_space;
  cv::Size m_padding;
  std::vector<cv::Mat> m_filters;
  std::vector<cv::Mat> m_flipped_filters;
};

#endif

