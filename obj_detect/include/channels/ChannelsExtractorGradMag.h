
/** ------------------------------------------------------------------------
 *
 *  @brief Channel feature extractors for LUV color space.
 *  @author Jorge Vela
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2019/07/08
 *
 *  ------------------------------------------------------------------------ */
//ChanExtractorMex.h

#ifndef CHANNELS_GRADMAG
#define CHANNELS_GRADMAG

#include <opencv/cv.hpp>
#include <vector>


class GradMagExtractor
{
public:
  GradMagExtractor
    (
    ){};

    void test();

    
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
