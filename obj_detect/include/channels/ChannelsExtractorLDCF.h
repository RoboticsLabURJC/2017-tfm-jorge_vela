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
#include <detectors/ClassifierConfig.h>
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
    * @param clf is the configuration of the detector
    * @param acf_impl_type See ChannelsExtractorACF for the allowed types.
    */
  ChannelsExtractorLDCF
    (
      std::vector<cv::Mat> filters,
      ClassifierConfig clf,
      std::string acf_impl_type = "pdollar"
    );

  /**
   * This method computes all the Piotr Dollar's Locally Decorrelated Channel Features
   * using the Aggregated Channels Features as cv::Mat with an input image:
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
      cv::Mat img // Should be a LUV image!!
    );

  /**
   * This method computes all the Piotr Dollar's Locally Decorrelated Channel Features
   * from the Aggregated Channels Features already computed
   *
   * @param src input image
   * @return std::vector<cv::Mat> vector with all the channels as cv:Mat.
   */
  std::vector<cv::Mat>
  extractFeaturesFromACF
    (
      const std::vector<cv::Mat>& acf_features
    );

  /**
   * This method computes all the Piotr Dollar's Locally Decorrelated Channel Features
   * using the Aggregated Channels Features as cv::Mat with an input image:
   *   - 3 color chanels in the LUV color space
   *   - 1 Gradient Magnitude channel
   *   - 6 HOG channels (6 orientations).
   *
   * @param src input image as a UMat
   * @return std::vector<cv::UMat> vector with all the channels as cv:UMat.
   */
  std::vector<cv::UMat>
  extractFeatures
    (
      cv::UMat img // Should be a LUV image!!
    );

  /**
   * This method computes all the Piotr Dollar's Locally Decorrelated Channel Features
   * from the Aggregated Channels Features already computed
   *
   * @param src input image as a UMat
   * @return std::vector<cv::UMat> vector with all the channels as cv:Mat.
   */
  std::vector<cv::UMat>
  extractFeaturesFromACF
    (
    const std::vector<cv::UMat>& acf_features
    );

private:
  std::string m_acf_impl_type;
  int m_shrink;
  std::string m_color_space;
  cv::Size m_padding;
  std::vector<cv::Mat> m_filters;
  std::vector<cv::Mat> m_flipped_filters;
  std::vector<cv::UMat> m_filters_umat;
  std::vector<cv::UMat> m_flipped_filters_umat;

  int m_gradientMag_normRad;
  float m_gradientMag_normConst;

  int m_gradientHist_binSize;
  int m_gradientHist_nOrients;
  int m_gradientHist_softBin;
  int m_gradientHist_full;

  ClassifierConfig m_clf;

};

#endif

