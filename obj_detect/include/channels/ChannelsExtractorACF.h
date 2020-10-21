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
#include <detectors/ClassifierConfig.h>
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
    * @param padding
    * @param shrink
    */
  ChannelsExtractorACF
    (
    ClassifierConfig clf,
    bool postprocess_channels = true,
    std::string impl_type = "opencv"
//    std::string impl_type = "pdollar"
    )
  {
    m_impl_type = impl_type;
    m_clf = clf;
    m_postprocess_channels = postprocess_channels;
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

  void
  postProcessChannels
    (
    const std::vector<cv::Mat>& acf_channels_no_postprocessed, // input
    std::vector<cv::Mat>& postprocessedChannels // output
    );

  cv::Mat 
  processChannels
    (
    cv::Mat img,
    cv::BorderTypes,
    int x,
    int y
    );

  int getNumChannels() { return 10; }

private:
  std::string m_impl_type;

  int m_shrink;
  std::string m_color_space;
  cv::Size m_padding;
  bool m_postprocess_channels;

  int m_gradientMag_normRad;
  float m_gradientMag_normConst;

  int m_gradientHist_binSize;
  int m_gradientHist_nOrients;
  int m_gradientHist_softBin;
  int m_gradientHist_full;

  ClassifierConfig m_clf;
};

#endif

