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
    cv::Size padding,
    int shrink
  )
{
    m_padding = padding;
    m_shrink = shrink;
    m_filters = filters;

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
  cv::Mat img // Should be a LUV image!!
  )
{
  // Extract the ACF channels
  ChannelsExtractorACF acfExtractor(m_shrink, "LUV");
  std::vector<cv::Mat> acf_channels = acfExtractor.extractFeatures(img);

  // Preprocessing of the ACF channels
  std::vector<cv::Mat> preprocessedChannels;
  int x = m_padding.width / m_shrink;
  int y = m_padding.height / m_shrink;

  for (cv::Mat c: acf_channels)
  {
    cv::Mat c_padded;
    c_padded = convTri(c, 1);
    copyMakeBorder( c_padded, c_padded, y, y, x, x, cv::BORDER_REFLECT, 0 );
    preprocessedChannels.push_back(c_padded);
  }

  if (m_filters.empty())
  {
    return preprocessedChannels; // Returning ACF channels after preprocessing
  }

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

      // NOTE: filter2D is not making real convolucion as conv2 in matlab (it implements correlation).
      // Thus we have to flip the kernel and change the anchor point. We have already flipped the filters
      // when we loaded them!!
      filter2D( preprocessedChannels[i], out_image, CV_32FC1 ,
                m_flipped_filters[i+(num_acf_channels*j)],
                cv::Point( -1,-1 ), 0, cv::BORDER_CONSTANT );
      out_image = ImgResample(out_image, round(0.5*out_image.size().width), round(0.5*out_image.size().height));
      ldcf_channels.push_back(out_image);
    }
  }

  return ldcf_channels;
}

