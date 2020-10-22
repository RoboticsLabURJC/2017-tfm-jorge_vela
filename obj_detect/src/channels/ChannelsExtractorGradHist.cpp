/** ------------------------------------------------------------------------
 *
 *  @brief Implementation of Channel extractor factory for Gradient Magnitude
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2020/10/22
 *
 *  ------------------------------------------------------------------------ */

#include <channels/ChannelsExtractorGradHist.h>
#include <channels/ChannelsExtractorGradHistOpenCV.h>
#include <channels/ChannelsExtractorGradHistPDollar.h>

std::shared_ptr<ChannelsExtractorGradHist>
ChannelsExtractorGradHist::createExtractor
  (
  std::string extractor_type,
  int binSize,
  int nOrients,
  int softBin,
  int full
  )
{
  std::shared_ptr<ChannelsExtractorGradHist> pExtractor;
  if (extractor_type == "opencv")
  {
    pExtractor.reset(new ChannelsExtractorGradHistOpenCV(binSize, nOrients, softBin, full));
  }
  else // if (extractor_type == "pdollar")
  {
    pExtractor.reset(new ChannelsExtractorGradHistPDollar(binSize, nOrients, softBin, full));
  }

  return pExtractor;
}
