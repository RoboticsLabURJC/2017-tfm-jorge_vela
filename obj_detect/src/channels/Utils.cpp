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





productChnsCompute channelsCompute(cv::Mat src, int shrink){

	productChnsCompute productCompute;

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

	h = imageCropped.size().height;
	w = imageCropped.size().width;


	std::vector<cv::Mat> luvImage = channExtract.extractFeatures(imageCropped); //IMAGENES ESCALA DE GRISES??

	cv::Mat dst;
	luvImage[0].copyTo(dst);

	cv::Mat dst2;
	luvImage[2].copyTo(dst2);

	//dst2.copyTo(luvImage[0]);
	//dst.copyTo(luvImage[2]);
	
	cv::Mat luv_image;
	merge(luvImage, luv_image);


	int size = imageCropped.cols*imageCropped.rows*dChan;
	float *M = new float[size](); // (size, sizeData, misalign)??
	float *O = new float[size]();


	printf("%d %d \n", w, h);
	gradMagExtract.gradMAdv(luv_image*255,M,O);

	printf("M: %.4f %.4f\n",  M[0], M[1]);
	printf("O: %.4f %.4f\n",  O[0], O[1]);

	cv::Mat dummy_query = cv::Mat(w,h,  CV_32F, M);

	cv::Mat M_to_img = convTri(dummy_query, 5);

	cv::Mat newM;
    M_to_img.convertTo(newM, CV_32F);
    float *dataM = newM.ptr<float>();

	gradMagExtract.gradMagNorm(M, dataM, w,h, 0.005);

	printf("M 0 modificada: %f \n", M[0]);
	printf("M 1 modificada: %f \n", M[1]);
	//for(int y=0;y<h;y++){ for(int x=0;x<w;x++) printf("%f ",M[x*w+y]); printf("\n");}
    //for(int y=0;y<h;y++){ for(int x=0;x<w;x++) printf("%.4f ",M[x*h+y]); printf("\n");}
	int h2 = h/4;
	int w2 = w/4;
	int sizeH = h2*w2*dChan*6;
	float *H = new float[sizeH]();

	gradHistExtract.gradHAdv(luv_image, M, O, H);
	printf("%f \n", H[0]);
	printf("%f \n", H[1]);
	printf("%f \n", H[2]);

	productCompute.image = luv_image;
	productCompute.M = M;
	productCompute.O = O;
	productCompute.H = H;

	return productCompute;
}



void getScales(	int nPerOct, int nOctUp, int minDs[], int shrink, int sz[]){
	if(sz[0]==0 || sz[1]==0){
		int scales[0];
		int scaleshw[0];
	}

	float val1 = (float)sz[0]/(float)minDs[0];
	float val2 = (float)sz[1]/(float)minDs[1];

	printf("%f %f \n", val1, val2 );

	float min = std::min(val1, val2);
	int nScales = floor(nPerOct*(nOctUp+log2(min))+1);
	printf("minValue %.4f %d \n", min, nScales);

	int d0 = 0;
	int d1 = 0;
	if(sz[0] < sz[1]){
		int d0 = sz[0];
		int d1 = sz[1];
	}else{
		int d0 = sz[1];
		int d1 = sz[0];
	}

	int scales[nScales];
	int scaleshw[nScales];


	for(int s = 0; s < nScales; s++){
		float s0=(round(d0*s/shrink)*shrink-.25*shrink)/d0;
		float s1=(round(d0*s/shrink)*shrink+.25*shrink)/d0;
	}
}










