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
#include <math.h>

using namespace cv;

cv::Mat ImgResample(cv::Mat src, int width, int height, int nChannels){
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
  for(i = 1; i <= kernel_size + 1; i++)
  {
    arrayKernel[i-1] = (float)i / valReduce;
  }

  int downCount = 0;
  for(int j = kernel_size; j > 0; j--)
  {
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

  int smooth = 1;
  ChannelsLUVExtractor channExtract{false, smooth};
  GradMagExtractor gradMagExtract;
  GradHistExtractor gradHistExtract;


  int dChan = src.channels();
  int h = src.size().height;
  int w = src.size().width;

  /*int crop_h = h % shrink;
  int crop_w = w % shrink;

  h = h - crop_h;
  w = w - crop_w;
	
  Rect cropImage = Rect(0,0,w, h);
  cv::Mat imageCropped = src(cropImage);*/

  std::vector<cv::Mat> luvImage = channExtract.extractFeatures(src); //IMAGENES ESCALA DE GRISES??

  cv::Mat dst;
  luvImage[0].copyTo(dst);

  cv::Mat dst2;
  luvImage[2].copyTo(dst2);

  luvImage[0] = dst2;
  luvImage[2] = dst;

  cv::Mat luv_image;
  merge(luvImage, luv_image);

  luv_image = convTri(luv_image, smooth);

  int size = src.cols*src.rows*dChan;
  float *M = new float[size](); 
  float *O = new float[size]();

  gradMagExtract.gradMAdv(luv_image*255,M,O,5);

  int h2 = h/4;
  int w2 = w/4;
  int sizeH = h2*w2*dChan*6;
  float *H = new float[sizeH]();

  gradHistExtract.gradH(luv_image, M, O, H);


  /*for(int i = 0; i < 14*17; i++){
    printf("%.4f \n", H[i]);
  }*/



  productCompute.image = luv_image;
  productCompute.M = M;
  productCompute.O = O;
  productCompute.H = H;

  return productCompute;
}


void getScales(	int nPerOct, int nOctUp, int minDs[], int shrink, int sz[]){
  if(sz[0]==0 || sz[1]==0)
  {
    int scales[0];
    int scaleshw[0];
  }
  
  float val1 = (float)sz[0]/(float)minDs[0];  
  float val2 = (float)sz[1]/(float)minDs[1];

  printf("%f %f \n", val1, val2 );

  float min = std::min(val1, val2);
  int nScales = floor(nPerOct*(nOctUp+log2(min))+1);
  printf("minValue %.4f %d \n", min, nScales);

  float d0 = 0;
  float d1 = 0;
  if(sz[0] < sz[1])
  {
    d0 = (float)sz[0];
    d1 = (float)sz[1];
  }
  else
  {
    d0 = (float)sz[1];
    d1 = (float)sz[0];
  }

	printf("%f %f\n", d0, d1);
	//int scales[nScales];
	//int scaleshw[nScales];

	
  float s = (-float(0-1)/float(nPerOct+nOctUp));
  float p = pow(2,s);
	
  for (int scale = 0; scale < 100/*abs(nScales)*/; scale++){
    printf("-----------------------------------------------------------\n");
    float valueIn = (-(float(scale))/float(nPerOct)+float(nOctUp));
    float s = pow(2,valueIn);

    float srk = (float)shrink;
    float s0=(round(d0*s/srk)*srk-.25*srk)/d0;
    float s1=(round(d0*s/srk)*srk+.25*srk)/d0;
		//printf("%f -- %f -- %f -- %f \n",valueIn, s, s0, s1);

    printf("%f %f %f %f \n", d0, s, srk , s0);

    float ss = 0.0;
    //float arr_es0[100];
    //float arr_es1[100];

    float xMin = 999999.99999;
    int pos = 0;
    float arrayPositions[1]; //1 se sustituirá por abs(nScales)
    while(ss < 100){
      float ss_mod = ss*(s1 - s0)+s0;
      float es0=d0*ss_mod; es0=abs(es0-round(es0/shrink)*shrink);
      //arr_es0[(int)(ss*100)] = es0;

      float es1=d1*ss_mod; es1=abs(es1-round(es1/shrink)*shrink);
      //arr_es0[(int)(ss*100)] = es1;
      ss = ss + 0.01;

      float x = max(es0, es1); 
      //arr_es0[(int)(ss*100)] = x;
      if(x < xMin)
      {
        xMin = x;
        pos = (int)(ss*100);
      }
    }
    arrayPositions[scale] = pos*0.01;
    printf("%f %d \n", xMin, pos ); //FUNCION MIN DE MATLAB RETORNA EL VALOR Y LA POSICION DEL ARRAY EN LA QUE SE ENCUENTRA
  }
}


//for(int s = 0; s > nScales; s--){
//float s = (-(0-1)/nPerOct+nOctUp);
//float s0=(round(d0*s/shrink)*shrink-.25*shrink)/d0;
//float s1=(round(d0*s/shrink)*shrink+.25*shrink)/d0;
//}







