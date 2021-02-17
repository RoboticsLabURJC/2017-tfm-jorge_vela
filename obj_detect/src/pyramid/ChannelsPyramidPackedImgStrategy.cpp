
#include <pyramid/ChannelsPyramidPackedImgStrategy.h>
#include <detectors/ClassifierConfig.h>
#include <channels/Utils.h>
#include <channels/ChannelsExtractorLDCF.h>
#include <opencv2/opencv.hpp>
#include <channels/Utils.h>
#include <cmath>
#include <iostream>
#include <algorithm>

#undef DEBUG
//#define DEBUG

std::vector<std::vector<cv::Mat>>
ChannelsPyramidPackedImgStrategy::compute
  (
  cv::Mat img,
  std::vector<cv::Mat> filters,
  std::vector<double>& scales,
  std::vector<cv::Size2d>& scaleshw,
  ClassifierConfig clf
  )
{
  cv::Size sz = img.size();
  getScales(clf.nPerOct, clf.nOctUp, clf.minDs, clf.shrink, sz, scales, scaleshw);

#ifdef DEBUG
  std::cout << "--> scales = ";
  for (uint i=0; i < scales.size(); i++)
  {
    std::cout << scales[i] << ", ";
  }
  std::cout << std::endl;
#endif

  int nScales = static_cast<int>(scales.size());
  ChannelsExtractorLDCF ldcfExtractor(filters, clf, m_channels_impl_type);

  // Setup the ROIs within the packed image of every image in the pyramid
  std::vector<cv::Size> pyramid_imgs_sizes;
  std::vector<cv::Size> pyramid_chns_sizes;
  for (int i=0; i< nScales; i++)
  {
    double s = scales[i];
    pyramid_imgs_sizes.emplace_back(round((sz.width * s) / clf.shrink) * clf.shrink,
                                    round((sz.height * s) / clf.shrink) * clf.shrink);

    pyramid_chns_sizes.emplace_back(round(0.5*pyramid_imgs_sizes[i].width / clf.shrink),
                                    round(0.5*pyramid_imgs_sizes[i].height / clf.shrink));

  }

  // Get the ROIs of a big packed image to put the resized images into
  std::vector<cv::Rect2i> pyr_imgs_rois = computePackedPyramidImageROIs(pyramid_imgs_sizes);
  std::vector<cv::Rect2i> pyr_chns_rois = computePackedPyramidImageROIs(pyramid_chns_sizes);

  // Create the packed image with filled with zeros.
  cv::Size packed_img_size = computePackedImageSize(pyr_imgs_rois);
  cv::Mat packed_img = cv::Mat::ones(packed_img_size, img.type());
  packed_img *= 128;

#ifdef DEBUG
  std::cout << "packed_img_size = " << packed_img_size << std::endl;
  int kk=0;
  std::cout << "======================" << std::endl;
  for (auto r: pyr_imgs_rois)
  {
    std::cout << "pyr_imgs_rois[" << kk << "] = " << r << std::endl;
    kk++;
  }
#endif

  // Resize the input image to the scales
  for (auto roi: pyr_imgs_rois)
  {
    cv::Mat roi_img = cv::Mat(packed_img, roi);
    cv::resize(img, roi_img, cv::Size(roi.width, roi.height), 0, 0, cv::INTER_AREA);
  }

  // We only need to compute the channels over the packed image
  std::vector<cv::Mat> chnsCompute = ldcfExtractor.extractFeatures(packed_img);

#ifdef DEBUG
  cv::imshow("packed image", packed_img);
  int i=0;
  for (auto chn: chnsCompute)
  {
    std::string s;
    s += i;
    cv::imshow(s, chn);
  }
  cv::waitKey();
#endif

  std::vector<std::vector<cv::Mat>> chnsPyramidData_out(nScales);
  for (int i=0; i < nScales; i++)
  {
    std::vector<cv::Mat> chnsPyramidData_out_i(chnsCompute.size());
    for (int j=0; j < chnsCompute.size(); j++)
    {
#ifdef DEBUG
      cv::imshow("results", chnsCompute[j]);
      cv::waitKey();
#endif
      cv::Mat chnROI = cv::Mat(chnsCompute[j], pyr_chns_rois[i]);
#ifdef DEBUG
      cv::imshow("chnROI", chnROI);
      cv::waitKey();
#endif
      chnROI.copyTo(chnsPyramidData_out_i[j]);
    }
    chnsPyramidData_out[i] = chnsPyramidData_out_i;
  }

  return chnsPyramidData_out;
}

// Adapted from: https://github.com/hunter-packages/ogles_gpgpu/blob/hunter/ogles_gpgpu/common/proc/pyramid.cpp
cv::Size
ChannelsPyramidPackedImgStrategy::computePackedImageSize
  (
  const std::vector<cv::Rect2i>& pyr_imgs_rois
  )
{
  int width = 0;
  int height = 0;

//  pyr_imgs_rois = computePackedPyramidImageROIs(pyramidImgsSizes);
  for (const auto& c : pyr_imgs_rois)
  {
    width = std::max(width, c.x + c.width);
    height = std::max(height, c.y + c.height);
  }

  return cv::Size(width, height);
}

// Adapted from: https://github.com/hunter-packages/ogles_gpgpu/blob/hunter/ogles_gpgpu/common/proc/pyramid.cpp
std::vector<cv::Rect2i>
ChannelsPyramidPackedImgStrategy::computePackedPyramidImageROIs
  (
  const std::vector<cv::Size>& pyramid_imgs_size
  )
{
  std::vector<cv::Rect2i> packed;

  int x = 0, y = 0;
  bool half = false;

  // Decrease going forward:
  int i = 0;
  for (; (i < pyramid_imgs_size.size()) && !half; i++)
  {
    packed.emplace_back(x, y, pyramid_imgs_size[i].width, pyramid_imgs_size[i].height);
    x += pyramid_imgs_size[i].width;
    if (pyramid_imgs_size[i].height * 2 < pyramid_imgs_size[0].height)
    {
      half = true;
    }
  }

  // Decrease going backward -- now x becomes the right edge
  int t, l, r, b = pyramid_imgs_size[0].height;
  for (; i < pyramid_imgs_size.size(); i++)
  {
    r = x;
    l = r - pyramid_imgs_size[i].width;
    t = b - pyramid_imgs_size[i].height;
    packed.emplace_back(l, t, pyramid_imgs_size[i].width, pyramid_imgs_size[i].height);
    x -= pyramid_imgs_size[i].width;
  }

  return packed;
}
