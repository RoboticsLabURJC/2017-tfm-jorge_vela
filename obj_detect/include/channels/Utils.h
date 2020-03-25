/** ------------------------------------------------------------------------
 *
 *  @brief Channel Utils.
 *  @author Jorge Vela
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2020/17/02
 *
 *  ------------------------------------------------------------------------ */

#ifndef IMAGE_UTILS
#define IMAGE_UTILS


#include <opencv/cv.hpp>
#include <vector>


cv::Mat ImgResample
(
	cv::Mat src, 
	int width,
	int height,
	int nChannels
);


void channelsCompute
(
	cv::Mat src,
	int shrink
);

cv::Mat convTri
(
	cv::Mat input_image,
	int kernel_size
);
#endif