/** ------------------------------------------------------------------------
 *
 *  @brief Implementation of Channel extractor factory for Gradient Magnitude
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2020/10/22
 *
 *  ------------------------------------------------------------------------ */

#include <channels/ChannelsExtractorGradMag.h>
#include <channels/ChannelsExtractorGradMagOpenCV.h>
#include <channels/ChannelsExtractorGradMagOpenCL.h>
#include <channels/ChannelsExtractorGradMagPDollar.h>

std::shared_ptr<ChannelsExtractorGradMag>
ChannelsExtractorGradMag::createExtractor
  (
  std::string extractor_type,
  int normRad,
  float normConst
  )
{
  std::shared_ptr<ChannelsExtractorGradMag> pExtractor;
  if (extractor_type == "opencv")
  {
    pExtractor.reset(new ChannelsExtractorGradMagOpenCV(normRad, normConst));
  }
  else if (extractor_type == "opencl")
  {
    pExtractor.reset(new ChannelsExtractorGradMagOpenCL(normRad, normConst));
  }
  else // if (extractor_type == "pdollar")
  {
    pExtractor.reset(new ChannelsExtractorGradMagPDollar(normRad, normConst));
  }

  return pExtractor;
}
