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





std::vector<cv::Mat> channelsCompute(cv::Mat src, int shrink){

  productChnsCompute productCompute;

  int smooth = 1;
  ChannelsLUVExtractor channExtract{false, smooth};
  GradMagExtractor gradMagExtract{5};
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

  /*cv::Mat dst;
  luvImage[0].copyTo(dst);

  cv::Mat dst2;
  luvImage[2].copyTo(dst2);

  luvImage[0] = dst2;
  luvImage[2] = dst;*/

  cv::Mat luv_image;
  merge(luvImage, luv_image);

  luv_image = convTri(luv_image, smooth);

  std::vector<cv::Mat> gMagOrient = gradMagExtract.extractFeatures(luv_image*255);

  std::vector<cv::Mat> gMagHist = gradHistExtract.extractFeatures(luv_image, gMagOrient);


  int lenVector = luvImage.size() +  gMagOrient.size() +  gMagHist.size();

  printf("%d\n", lenVector);
  std::vector<cv::Mat> chnsCompute(lenVector);

  for(int i=0; i<3; i++)
    chnsCompute[i] = luvImage[i];

  for(int i=0; i<1; i++)
    chnsCompute[i+3] = gMagOrient[i]; //Meter solo la magnitud, no añadir la orientación

  for(int i =0; i <  gMagHist.size(); i++) 
    chnsCompute[i+4] = gMagHist[i];

  return chnsCompute;
}



void chnsPyramid(cv::Mat img){
  printf("channelsPyramids\n");

  int smooth = 1;
  ChannelsLUVExtractor channExtract{false, smooth};


  //CONVERT I TO APPROPIATE COLOR SPACE-------------------------------------
  std::vector<cv::Mat> luvImage = channExtract.extractFeatures(img); //IMAGENES ESCALA DE GRISES??
  cv::Mat luv_image;
  merge(luvImage, luv_image);
  //EN ESTAS LINEAS EL COMPRUEBA QUE HA HECHO BIEN LA CONVERSION DE COLORES...?

  //-------------------------------------------------------------------------

  int nOctUp=0;
  int nPerOct=8;
  int nApprox = -1;
  int sz;
  int shrink = 4;

  //GET SCALES AT WHICH TO COMPUTE FEATURES---------------------------------
  std::vector<int> scales, scaleshs;
  
  int nScales = scales.size();
  int isR;
  if(1){
    isR = 1;
  }else{
    isR = 1+nOctUp*nPerOct;
  }
 
  int sizeisR = nScales/(nApprox+1);
  std::vector<int> isRarr;

  
  int iLoop = 0;
  while(iLoop<sizeisR){
    isRarr.push_back(1 + iLoop*(nApprox+1)); //(0) =  1 + iLoop*(nApprox+1);
    //iLoop = iLoop + nApprox+1;
  }

 
  std::vector<int> arrJ;
  arrJ.push_back(0); //[0] = 0;
  for(int i = 0; i < isRarr.size() - 1; i++){
    arrJ.push_back((isRarr[i] + isRarr[i+1])/2);//[i+1] = (isRarr[i] + isRarr[i+1])/2;
  }
  arrJ.push_back(nScales); //[sizeisR+1]= nScales;
  

  std::vector<int> isN;//[nScales];

  for(int i=0; i< isRarr.size(); i++){
    int val1 = arrJ[i];
    int val2 = arrJ[i+1];
    int lenghtArr = val2 - val1;
    std::vector<int> isNval;//[lenghtArr];
    for(int j = val1; j = val2; j++){
      isNval.push_back(j); //[j-vala1] = j;
    }
    isN.push_back(lenghtArr); //[i] = isNval;
  }
  //-------------------------------------------------------------------------

  //COMPUTE IMAGE PYRAMID----------------------------------------------------
  for(int i=0; i<isRarr.size(); i++){
    int s = scales[i];int sz1=round(sz*s/shrink)*shrink;

    cv::Mat I1;
    if(sz==sz1){
      I1 = img;
    } else {
      I1=ImgResample(img,sz1,sz1,1);
    }

    if(s==.5 && (nApprox>0 || nPerOct==1)){
      img = I1;
    }
    channelsCompute(I1,shrink);
  }

  //-------------------------------------------------------------------------

  //if lambdas not specified compute image specific lambdas.. SE OBVIA------

  //SUPONEMOS QUE NO SE DARÁ ESTE CASO---------------------------------------


  //COMPUTE IMAGE PYRAMID [APPROXIMATE SCALES]-------------------------------
  std::vector<int> isA; //MODIFICAR
  for(int i=0; i < isA.size(); i++){
    //isRarr = isN[i];
    //int sz1=round(sz*scales[i]/shrink)*shrink;

    printf("hoola %d\n",i );
  }

  //-------------------------------------------------------------------------

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

  //float min = std::min(val1, val2);
  int nScales = floor(nPerOct*(nOctUp+log2(std::min(val1, val2)))+1);
  //printf("nScales %d \n", nScales);

  std::vector<float> scales;
  for(int i = 0; i< nScales; i++){
    float scalesValFloat = (-(float(i)/float(nPerOct))+float(nOctUp));
    float scalesVal = pow(2,scalesValFloat);
    scales.push_back(scalesVal);
  }

  printf("%d\n", (int)scales.size());

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

  for (int i = 0; i <  nScales; i ++){
    float s = scales[i];

    float s0=(round(d0*s/ (float)shrink)* (float)shrink-.25* (float)shrink)/d0;
    float s1=(round(d0*s/ (float)shrink)* (float)shrink+.25* (float)shrink)/d0;

    //printf("s0: %.4f s1:%.4f \n",s0, s1 );
    float ssLoop = 0.0;
    //float xMin = 999999.99999; //AÑADIR LIMITS
    //int pos = 0;
    std::vector<float> arrayPositions; //1 se sustituirá por abs(nScales)
   
    std::vector<float> ss;
    std::vector<float> es0;
    std::vector<float> es1;

    while(ssLoop < 1){
      float ssVal = ssLoop*(s1-s0)+s0;
      ss.push_back(ssVal);
      float es0val = d0*ssVal;  es0val=abs(es0val-round(es0val/shrink)*shrink);
      float es1val = d1*ssVal;  es1val=abs(es1val-round(es1val/shrink)*shrink);

      es0.push_back(es0val);
      es1.push_back(es1val);

      //printf("Valores --> %.4f %.4f %.4f \n", es0val, es1val, ssVal);

      ssLoop = ssLoop + 0.01;

    }
    std::vector<float> x = max(es0, es1);
    int pos = 0;
    float xMin = 999999.99999; 
    for(int i=0; i < x.size(); i++){
      if( x[i] < xMin){
        xMin = x[i];
        pos = i;
      }
    }
    scales[i] = ss[pos];
  }

  /*for(int i=0; i< scales.size(); i++){
    printf("scale: %.4f\n", scales[i] );
  }*/

  std::vector<float> kp;
  std::vector<float> scales2;
  for(int i = 0; i < scales.size()-1; i++){
    //printf("--> %.4f %.4f\n",scales[i],  scales[i+1] );
    int kpVal = (scales[i] != scales[i+1]);
    if(kpVal == 1){
      //printf("%.4f\n",scales[i] );
      scales2.push_back(scales[i]);
    }
    kp.push_back(kpVal);
  }
  scales2.push_back(scales[scales.size()-1]);


  //std::vector<float> scaleshw;
  int sizeScaleshw = scales2.size();
  float scaleshw[sizeScaleshw][sizeScaleshw];
  for(int i = 0; i < sizeScaleshw; i++){
    float h = round(sz[0]*scales2[i]/shrink)*shrink/sz[0];
    float w = round(sz[1]*scales2[i]/shrink)*shrink/sz[1];
    scaleshw[i][0] = h;
    scaleshw[i][1] = w;
  }
  for(int i =0; i < sizeScaleshw; i++){
    printf("%.4f %.4f\n", scaleshw[i][0], scaleshw[i][1]);
  }
}







