
/** ------------------------------------------------------------------------
 *
 *  @brief Channel feature extractors for LUV color space.
 *  @author Jorge Vela
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2019/07/08
 *
 *  ------------------------------------------------------------------------ */

#ifndef CHANNELS_GRADHIST
#define CHANNELS_GRADHIST

#include <opencv/cv.hpp>
#include <vector>


class GradHistExtractor
{

public:
  GradHistExtractor
    (
    	int binSize = 8,
    	int nOrients = 8,
    	int softBin = 1,
    	int full = 0
    ) {m_binSize = binSize;
       m_nOrients = nOrients;
       m_softBin = softBin;
       m_full = full;};

  float* allocW
  (
  	int size, 
  	int sf,
  	int misalign
  );

  void gradH
    (
    	float* M,
    	float* O,
    	float* H
    );

  void gradHAdv
    (
      cv::Mat image,
      float* M,
      float* O,
      float* H
    );

private:
	int m_binSize;
	int m_nOrients;
	int m_softBin;
	int m_full;

protected:
	void gradHist(
		float *M, 
		float *O, 
		float *H, 
		int h,
		int w,
  		int bin, 
  		int nOrients, 
  		int softBin, 
  		bool full
  		);

	void gradQuantize( 
		float *O, 
		float *M, 
		int *O0, 
		int *O1, 
		float *M0, 
		float *M1,
  		int nb, 
  		int n, 
  		float norm, 
  		int nOrients, 
  		bool full, 
  		bool interpolate );
};

#endif
