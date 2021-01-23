/** ------------------------------------------------------------------------
 *
 *  @brief Channel feature extractors for magnitude and orient gradients.
 *  @author Jorge Vela
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2019/07/08
 *
 *  ------------------------------------------------------------------------ */

#ifndef CHANNELS_EXTRACTOR_GRAD_MAG_OPENCL
#define CHANNELS_EXTRACTOR_GRAD_MAG_OPENCL

#include <channels/ChannelsExtractorGradMag.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <vector>

class ChannelsExtractorGradMagOpenCL: public ChannelsExtractorGradMag
{
public:
  ChannelsExtractorGradMagOpenCL
    (
      int normRad = 0,
      float normConst = 0.005
    ); /*: ChannelsExtractorGradMag(normRad, normConst)
  {};*/

  //virtual ~ChannelsExtractorGradMagOpenCL() {};
    
  virtual std::vector<cv::Mat> extractFeatures
    (
    cv::Mat img
    );

//  std::vector<cv::UMat> extractFeatures
//    (
//    cv::UMat img
//    );

  void
  gradMagOpenCL
    (
    const cv::UMat I, // one channel
    cv::UMat& M,
    cv::UMat& O,
    bool full = true
    );

  /*std::vector<cv::Mat> extractFeatures
    (
      cv::UMat img 
    );*/

protected:
    cv::UMat m_kernel_horiz;
    cv::UMat m_kernel_vert;

};

#endif
