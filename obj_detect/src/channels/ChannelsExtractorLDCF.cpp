/** ------------------------------------------------------------------------
 *
 *  @brief Implementation of Aggregated Channels Features (ACF)
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2020/09/25
 *
 *  ------------------------------------------------------------------------ */

#include <iostream>
#include <channels/ChannelsExtractorACF.h>
#include <channels/ChannelsExtractorLDCF.h>
#include <channels/Utils.h>
#include <opencv2/opencv.hpp>

ChannelsExtractorLDCF::ChannelsExtractorLDCF
  (
    std::vector<cv::Mat> filters,
    ClassifierConfig clf
    /*cv::Size padding,
    int shrink,
    int gradientMag_normRad,
    float gradientMag_normConst ,
    int gradientHist_binSize,
    int gradientHist_nOrients,
    int gradientHist_softBin, 
    int gradientHist_full */
  )
{ 
    m_filters = filters;  
    m_clf = clf;
    /*m_padding = padding;
    m_shrink = shrink;
    m_filters = filters;  

    m_gradientMag_normRad = gradientMag_normRad;
    m_gradientMag_normConst = gradientMag_normConst;

    m_gradientHist_binSize = gradientHist_binSize;
    m_gradientHist_nOrients = gradientHist_nOrients;
    m_gradientHist_softBin = gradientHist_softBin;
    m_gradientHist_full = gradientHist_full;*/
    for (cv::Mat f: m_filters)
    {
      // NOTE: filter2D is a correlation and to do convolution as in Matlab's conv2
      //       we have to flip the kernels in advance.
      //       We do it before using then.
      cv::Mat f_flipped;
      cv::flip(f, f_flipped, -1);
      m_flipped_filters.push_back(f);
    }
};

std::vector<cv::Mat> ChannelsExtractorLDCF::extractFeatures
  (
  cv::Mat img
  )
{
  // Extract the ACF channels
  ChannelsExtractorACF acfExtractor(m_clf, true);//(m_clf.padding, m_clf.shrink, true, m_clf.gradMag.normRad, m_clf.gradMag.normConst,m_clf.gradHist.binSize,m_clf.gradHist.nOrients,m_clf.gradHist.softBin,m_clf.gradHist.full);
  std::vector<cv::Mat> acf_channels = acfExtractor.extractFeatures(img);

  if (m_filters.empty())
  {
    return acf_channels; // Returning ACF channels after preprocessing
  }

  // Returning LDCF features (filtered ACF channels)
  return extractFeaturesFromACF(acf_channels);
}

std::vector<cv::Mat>
ChannelsExtractorLDCF::extractFeaturesFromACF
  (
    const std::vector<cv::Mat>& acf_channels
  )
{
    // Now use convolution over the preprocessed ACF channels with the LDCF filters.
    std::vector<cv::Mat> ldcf_channels;
    int num_filters_per_channel = m_flipped_filters.size() / acf_channels.size();
    assert(m_flipped_filters.size() % acf_channels.size() == 0);
    int num_acf_channels = acf_channels.size();
    for(int j = 0; j < num_filters_per_channel; j++)
    {
      for(int i = 0; i < num_acf_channels; i++)
      {
        cv::Mat out_image;

        // NOTE: filter2D is not making real convolution as conv2 in matlab (it implements correlation).
        // Thus we have to flip the kernel and change the anchor point. We have already flipped the filters
        // when added them to the constructor!!
        filter2D( acf_channels[i], out_image, CV_32FC1 ,
                  m_flipped_filters[i+(num_acf_channels*j)],
                  cv::Point( -1,-1 ), 0, cv::BORDER_CONSTANT );

        out_image = ImgResample(out_image, round(0.5*out_image.size().width), round(0.5*out_image.size().height));
        ldcf_channels.push_back(out_image);
      }
    }

    return ldcf_channels;
}

