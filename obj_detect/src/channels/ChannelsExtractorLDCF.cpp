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
    ClassifierConfig clf,
    std::string acf_impl_type
  )
{ 
    m_filters = filters;  
    m_clf = clf;
    for (cv::Mat f: m_filters)
    {
      // NOTE: filter2D is a correlation and to do convolution as in Matlab's conv2
      //       we have to flip the kernels in advance.
      //       We do it before using then.
      cv::Mat f_flipped;
      cv::flip(f, f_flipped, -1);
      m_flipped_filters.push_back(f); /// TODO: It f what it is pushed on the m_flipped_filters. Is this right?
      if (acf_impl_type == "opencl")
      {
        cv::UMat f_flipped_umat;
        f.copyTo(f_flipped_umat);
        m_flipped_filters_umat.push_back(f_flipped_umat);
      }
    }

    m_acf_impl_type = acf_impl_type;
};

std::vector<cv::Mat>
ChannelsExtractorLDCF::extractFeatures
  (
  cv::Mat img
  )
{
  // Extract the ACF channels
  ChannelsExtractorACF acfExtractor(m_clf, true, m_acf_impl_type);
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

      // NOTE: filter2D is not making real convolution as conv2 in matlab, it performs correlation.
      // Thus, we have to flip the kernel and change the anchor point. We have already flipped the filters
      // when added them to the constructor!!
      filter2D( acf_channels[i], out_image, CV_32FC1 ,
                m_flipped_filters[i+(num_acf_channels*j)],
                cv::Point( -1,-1 ), 0, cv::BORDER_CONSTANT );

//      out_image = ImgResample(out_image, round(0.5*out_image.size().width), round(0.5*out_image.size().height));
      ImgResample(out_image, out_image, round(0.5*out_image.size().width), round(0.5*out_image.size().height));
      ldcf_channels.push_back(out_image);
    }
  }

  return ldcf_channels;
}

std::vector<cv::UMat>
ChannelsExtractorLDCF::extractFeatures
  (
  cv::UMat img // Should be a LUV image!!
  )
{
  // Extract the ACF channels
  ChannelsExtractorACF acfExtractor(m_clf, true, "opencl");
  std::vector<cv::UMat> acf_channels = acfExtractor.extractFeatures(img);

  if (m_flipped_filters_umat.empty())
  {
    return acf_channels; // Returning ACF channels after preprocessing
  }

  // Returning LDCF features (filtered ACF channels)
  return extractFeaturesFromACF(acf_channels);
}

std::vector<cv::UMat>
ChannelsExtractorLDCF::extractFeaturesFromACF
  (
  const std::vector<cv::UMat>& acf_channels
  )
{
  // Now use convolution over the preprocessed ACF channels with the LDCF filters.
  std::vector<cv::UMat> ldcf_channels;
  int num_filters_per_channel = m_flipped_filters.size() / acf_channels.size();
  assert(m_flipped_filters.size() % acf_channels.size() == 0);
  int num_acf_channels = acf_channels.size();
  for(int j = 0; j < num_filters_per_channel; j++)
  {
    for(int i = 0; i < num_acf_channels; i++)
    {
      cv::UMat out_image;

      // NOTE: filter2D is not making real convolution as conv2 in matlab, it performs correlation.
      // Thus, we have to flip the kernel and change the anchor point. We have already flipped the filters
      // when added them to the constructor!!
      filter2D( acf_channels[i], out_image, CV_32FC1 ,
                m_flipped_filters_umat[i+(num_acf_channels*j)],
                cv::Point( -1,-1 ), 0, cv::BORDER_CONSTANT );

//      out_image = ImgResample(out_image, round(0.5*out_image.size().width), round(0.5*out_image.size().height));
      ImgResample(out_image, out_image, round(0.5*out_image.size().width), round(0.5*out_image.size().height));
      ldcf_channels.push_back(out_image);
    }
  }

  return ldcf_channels;
}

