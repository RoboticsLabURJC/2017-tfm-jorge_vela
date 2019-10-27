
/** ------------------------------------------------------------------------
 *
 *  @brief Channel feature extractors for LUV color space.
 *  @author Jorge Vela
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2019/07/08
 *
 *  ------------------------------------------------------------------------ */

#ifndef CHANNELS_EXTRACTOR_MEX
#define CHANNELS_EXTRACTOR_MEX

#include <opencv/cv.hpp>
#include <vector>

class ChannelsMexExtractor
{
public:
  ChannelsMexExtractor
    (
    ) {};

  float* allocW
  (
  	int size, 
  	int sf,
  	int misalign
  );
  void gradM
    (
    	float* I,
    	float* M,
    	float* O
    );

};

#endif
