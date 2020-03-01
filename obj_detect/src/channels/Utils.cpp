/** ------------------------------------------------------------------------
 *
 *  @brief Channel Utils.
 *  @author Jorge Vela
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2020/17/02
 *
 *  ------------------------------------------------------------------------ */


#include <channels/ImageResample.h>
#include <channels/ChannelsExtractorLUV.h>
#include <channels/ChannelsExtractorGradMag.h>
#include <channels/ChannelsExtractorGradHist.h>

#include <opencv/cv.hpp>
#include <channels/Utils.h>

using namespace cv;

cv::Mat ImgResample(cv::Mat src, int width, int height, int nChannels){
	//cv::Mat dst;
	cv::Mat dst(height, width, CV_8UC3, Scalar(0, 0, 0));
	resize(src, dst, dst.size(), 0, 0, INTER_LINEAR);
	return dst;
}


void channelsCompute(cv::Mat src, int shrink){
	ChannelsLUVExtractor channExtract{false, 1};
  	GradMagExtractor gradMagExtract;
  	GradHistExtractor gradHistExtract;


  	int dChan = src.channels();
	int h = src.size().height;
	int w = src.size().width;

	int crop_h = h % shrink;
	int crop_w = w % shrink;

	//printf("%d %d %d %d \n",h,w, crop_h, crop_w );

	h = h - crop_h;
	w = w - crop_w;

	//imshow( "Display window A", src );

	Rect cropImage = Rect(0,0,w, h);
	cv::Mat imageCropped = src(cropImage);
	//imshow( "Display window B", imageCropped );
	//waitKey(0);

	std::vector<cv::Mat> channels = channExtract.extractFeatures(src);


	int size = src.cols*src.rows*dChan;
	float *M = new float[size](); // (size, sizeData, misalign)??
	float *O = new float[size]();

	int h2 = h/4;
	int w2 = w/4;
	int sizeH = h2*w2*dChan*6;


	float *H = new float[sizeH]();
	gradMagExtract.gradMAdv(src,M,O);
	gradHistExtract.gradH(M, O, H);

}