/** ------------------------------------------------------------------------
 *
 *  @brief Channel Utils.
 *  @author Jorge Vela
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2020/17/02
 *
 *  ------------------------------------------------------------------------ */


#include <channels/Utils.h>
#include <channels/ChannelsExtractorLUV.h>
#include <channels/ChannelsExtractorGradMag.h>
#include <channels/ChannelsExtractorGradHist.h>

#include <opencv/cv.hpp>
#include <channels/Utils.h>

using namespace cv;

cv::Mat ImgResample(cv::Mat src, int width, int height, int nChannels){
	//cv::Mat dst;
	cv::Mat dst(height, width, CV_32F, Scalar(0, 0, 0));
	resize(src, dst,Size(width,height), 0,0, INTER_LINEAR);

	transpose(dst, dst);

	return dst;
}


cv::Mat convTri(cv::Mat input_image, int kernel_size){

	cv::Mat output_image, help_image;

    cv::Point anchor;
    anchor = cv::Point( -1, -1 );

    float valReduce = (kernel_size + 1)*(kernel_size + 1);
    float arrayKernel[kernel_size*2 ];
    
    int i;
    for(i = 1; i <= kernel_size + 1; i++){
      arrayKernel[i-1] = (float)i / valReduce;
    }

    int downCount = 0;
    for(int j = kernel_size; j > 0; j--){
      arrayKernel[i-1] = (j - downCount) / valReduce;
      downCount = downCount++; 
      i = i+1;
    }

    cv::Mat kernel = cv::Mat((kernel_size*2)+1,1,  CV_32F, arrayKernel);
    filter2D(input_image, help_image, -1 , kernel, anchor, 0, cv::BORDER_REFLECT );
    kernel = cv::Mat(1,(kernel_size*2)+1,  CV_32F, arrayKernel);
    filter2D(help_image, output_image, -1 , kernel, anchor, 0, cv::BORDER_REFLECT );

    return output_image;
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

	h = h - crop_h;
	w = w - crop_w;


	Rect cropImage = Rect(0,0,w, h);
	cv::Mat imageCropped = src(cropImage);

	std::vector<cv::Mat> luvImage = channExtract.extractFeatures(src); //IMAGENES ESCALA DE GRISES??

	cv::Mat dst;
	luvImage[0].copyTo(dst);

	cv::Mat dst2;
	luvImage[2].copyTo(dst2);

	dst2.copyTo(luvImage[0]);
	dst.copyTo(luvImage[2]);
	
	cv::Mat luv_image;
	merge(luvImage, luv_image);


	int size = src.cols*src.rows*dChan;
	float *M = new float[size](); // (size, sizeData, misalign)??
	float *O = new float[size]();

	gradMagExtract.gradMAdv(luv_image*255,M,O);


	cv::Mat dummy_query = cv::Mat(w,h,  CV_32F, M);
	cv::Mat M_to_img = convTri(dummy_query, 5);

	cv::Mat newM;
    M_to_img.convertTo(newM, CV_32F);
    float *dataM = newM.ptr<float>();

	gradMagExtract.gradMagNorm(M, dataM, 69, 57, 0.005);

	int h2 = h/4;
	int w2 = w/4;
	int sizeH = h2*w2*dChan*6;
	float *H = new float[sizeH]();
	
	//gradHistExtract.gradHAdv(luv_image, M, O, H);

	/*printf("%.4f\n", H[0]);
	printf("%.4f\n", H[1]);
	printf("%.4f\n", H[2]);*/

}