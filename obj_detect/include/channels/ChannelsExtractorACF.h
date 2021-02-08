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
#include <channels/ChannelsExtractorLUV.h>
#include <channels/ChannelsExtractorGradMag.h>
#include <channels/ChannelsExtractorGradHist.h>
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
    * @param clf Configuration variables for ACF computation.
    * @param postprocess_channels postprocess or not the ACF channels (to be used in ChannelsPyramid).
    * @param impl_type By now the implementations are "pdollar" and "opencv".
    */
  ChannelsExtractorACF
    (
    ClassifierConfig clf,
    bool postprocess_channels = true,
    std::string impl_type = "pdollar"
    );

  /**
   * This method computes all the Piotr Dollar's Aggregated Channels Features as cv::Mat from an input image:
   *   - 3 color chanels in the LUV color space
   *   - 1 Gradient Magnitude channel
   *   - 6 HOG channels (6 orientations).
   * 
   * @param src input image
   * @return std::vector<cv::Mat> vector with all the channels as cv:Mat.
   */    
  std::vector<cv::Mat>
  extractFeatures
    (
    cv::Mat img
    );

  /**
   * This method computes all the Piotr Dollar's Aggregated Channels Features as cv::Mat from an input image:
   *   - 3 color chanels in the LUV color space
   *   - 1 Gradient Magnitude channel
   *   - 6 HOG channels (6 orientations).
   *
   * @param src input image as UMat
   * @return std::vector<cv::UMat> vector with all the channels as cv:Mat.
   */
  std::vector<cv::UMat>
  extractFeatures
    (
    cv::UMat img
    );

  int getNumChannels() { return 10; }

  void
  postProcessChannels
    (
    const std::vector<cv::Mat>& acf_channels_no_postprocessed, // input
    std::vector<cv::Mat>& postprocessedChannels // output
    );

  void
  postProcessChannels
    (
    const std::vector<cv::UMat>& acf_channels_no_postprocessed, // input
    std::vector<cv::UMat>& postprocessedChannels // output
    );


private:

  cv::Mat
  processChannels
    (
    cv::Mat img,
    cv::BorderTypes,
    int x,
    int y
    );


  cv::UMat
  processChannels
    (
    cv::UMat img,
    cv::BorderTypes,
    int x,
    int y
    );

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

  std::shared_ptr<ChannelsExtractorGradMag> m_pGradMagExtractor;
  std::shared_ptr<ChannelsExtractorGradHist> m_pGradHistExtractor;
  std::shared_ptr<ChannelsExtractorLUV> m_pLUVExtractor;
};

#endif

