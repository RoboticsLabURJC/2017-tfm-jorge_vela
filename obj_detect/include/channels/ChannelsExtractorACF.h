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
    * @param padding
    * @param shrink
    */
  ChannelsExtractorACF
    (
      cv::Size padding,
      int shrink,
      bool postprocess_channels = true,
      int gradientMag_normRad=0,
      float gradientMag_normConst = 0.005,
      int gradientHist_binSize = 8, //2
      int gradientHist_nOrients = 6, //6
      int gradientHist_softBin = 1,
      int gradientHist_full = 0,
      bool impl_type = "opencv"
    ) 
  {
      m_impl_type = impl_type;

      m_padding = padding;
      m_shrink = shrink;

      m_postprocess_channels = postprocess_channels;
      m_gradientMag_normRad = gradientMag_normRad;
      m_gradientMag_normConst = gradientMag_normConst;
      
      m_gradientHist_binSize = gradientHist_binSize;
      m_gradientHist_nOrients = gradientHist_nOrients;
      m_gradientHist_softBin = gradientHist_softBin;
      m_gradientHist_full = gradientHist_full;
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
};

#endif

