/** ------------------------------------------------------------------------
 *
 *  @brief Implementation of Channel feature extractors for LUV color space.
 *  @author Jorge Vela
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2020/17/02
 *
 *  ------------------------------------------------------------------------ */




#include <channels/ImageResample.h>
#include <opencv/cv.hpp>


using namespace cv;

cv::Mat ImageResample::ImgResample(cv::Mat src, int width, int height, int nChannels){
	//cv::Mat dst;
	cv::Mat dst(height, width, CV_8UC3, Scalar(0, 0, 0));
	resize(src, dst, dst.size(), 0, 0, INTER_LINEAR);
	return dst;
}