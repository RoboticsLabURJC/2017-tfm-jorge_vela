/** ------------------------------------------------------------------------
 *
 *  @brief Channel feature extractors for LUV color space.
 *  @author Jorge Vela
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2020/17/02
 *
 *  ------------------------------------------------------------------------ */



#ifndef IMAGE_RESAMPLE
#define IMAGE_RESAMPLE

#include <opencv/cv.hpp>
#include <vector>

class ImageResample
{
public:
	ImageResample
	(
	){};
	cv::Mat ImgResample
	(
		cv::Mat src, 
		int width,
		int height,
		int nChannels
	);
};

#endif
