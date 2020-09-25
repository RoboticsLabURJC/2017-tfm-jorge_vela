/** ------------------------------------------------------------------------
 *
 *  @brief Agregated Channel Features (ACF) Extractor
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2020/09/25
 *
 *  ------------------------------------------------------------------------ */

#ifndef CHANNELS_EXTRACTOR_ACF
#define CHANNELS_EXTRACTOR_ACF

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

/** ------------------------------------------------------------------------
 *
 *  @brief Class for ACF extraction: LUV, Gradient and HOG.
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2020/09/03
 *
 *  ------------------------------------------------------------------------ */
class ChannelsExtractorACF
{
public:
   /**
    * This constructor sets the parameters for computing the ACF features.
    *
    * @param shrink
    * @param color_space string describing the input image to extractFeatures color space.
    */
  ChannelsExtractorACF
    (
      int shrink,
      std::string color_space = "LUV"
    ) 
    {
      m_shrink = shrink;
      m_color_space = color_space;
    };

  /**
   * This method computes all the Piotr Dollar's Aggregated Channels Features as cv::Mat from an input image:
   *   - 3 color chanels in the LUV color space
   *   - 1 Gradient Magnitude channel
   *   - 6 HOG channels (6 orientations).
   * 
   * @param src input image
   * @return std::vector<cv::Mat> vector with all the channels as cv:Mat.
   */    
  std::vector<cv::Mat> extractFeatures
    (
      cv::Mat img
    );

private:
  int m_shrink;
  std::string m_color_space;

};

#endif

