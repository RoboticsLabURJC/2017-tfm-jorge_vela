/** ------------------------------------------------------------------------
 *
 *  @brief Implementation of Channel feature extractors for LUV color space.
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2020/10/22
 *
 *  ------------------------------------------------------------------------ */
#include <channels/ChannelsExtractorLUV.h>
//#include <channels/ChannelsExtractorLUVOpenCV.h>
#include <channels/ChannelsExtractorLUVPDollar.h>

std::shared_ptr<ChannelsExtractorLUV>
ChannelsExtractorLUV::createExtractor
  (
  std::string extractor_type,
  bool smooth,
  int smooth_kernel_size
  )
{
  std::shared_ptr<ChannelsExtractorLUV> pExtractor;
//  if (extractor_type == "opencv")
//  {
//    pExtractor.reset(new ChannelsExtractorLUVOpenCV(normRad, normConst));
//  }
//  else // if (extractor_type == "pdollar")
//  {
    pExtractor.reset(new ChannelsExtractorLUVPDollar(smooth, smooth_kernel_size));
//  }

  return pExtractor;
}
