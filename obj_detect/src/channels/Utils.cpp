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

#include <opencv2/opencv.hpp>
#include <channels/Utils.h>
#include <math.h>

//using namespace cv;

/**
 * Función Imgresample. Encargada de redimensionar una imagen de entrada, al tamaño de ancho y alto 
 * que se le pase por parámetros. 
 *
 * @param src: Imagen que se quiere redimensionar
 * @param width: Ancho de la imagen de salida
 * @param height: Alto de la imagen de salida
 * @param norm: [1] Valor por el que se multiplican los píxeles de salida
 * @return cv::Mat: Imagen redimensionada
 * 
 */
cv::Mat ImgResample(cv::Mat src, int width, int height, int norm){
  cv::Mat dst(height, width, CV_32F, cv::Scalar(0, 0, 0));
  resize(src, dst,cv::Size(width,height), 0,0, cv::INTER_AREA); //DICE QUE EN ALGUNOS CASOS NO UTILIZA ANTIALIASING OFF, POR LO QUE SERÍA INTER_AREA, EL CASO NORMAL ES INTER_LINEAR
  //dst = norm*dst;
  return dst;
}

/**
 * Funcion convTri. Convoluciona una imagen por un filtro de triangulo 2D. 
 *
 * @param input_image: Imagen de entrada la cual se quiere convolucionar.
 * @param kernel_size: Tamaño del kernel (radio) que se quiere para el filtro.
 *
 * @return cv::Mat: Imagen de retorno despues del filtro.
 */
cv::Mat convTri(cv::Mat input_image, int kernel_size){

  cv::Mat output_image, help_image;

  cv::Point anchor;
  anchor = cv::Point( -1, -1 ); //tipo de salida = tipo elementos imagen entrada, mirar este valor, CV_32F

  float valReduce = (kernel_size + 1)*(kernel_size + 1);
  float arrayKernel[kernel_size*2];
    
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
  double delta = 0;

  cv::Mat kernel = cv::Mat((kernel_size*2)+1,1,  CV_32FC1, arrayKernel); //
  filter2D(input_image, help_image, CV_32FC1 , kernel, anchor, delta, cv::BORDER_REFLECT );
  kernel = cv::Mat(1,(kernel_size*2)+1,  CV_32FC1, arrayKernel);
  filter2D(help_image, output_image, CV_32FC1 , kernel, anchor, delta, cv::BORDER_REFLECT );

  cv::Mat img3;
  output_image.convertTo(img3, CV_32FC1);    
/*
  float *valueM = img3.ptr<float>();
  printf("Convtri: \n");
  for(int i = 0; i < 15; i++)
    printf("%.4f ", valueM[i] );
  printf("\n");
  */
  return output_image;
}

struct productChnsCompute
  {
    cv::Mat image;
    float* M;
    float* O;
    float* H;
} ;

/**
 * Funcion channesCompute. Dada una imagen de entrada calcula las principales características
 * las cuales retorna como imagenes en un vector de cv::Mat. Los valores que retorna son:
 * (1) Canales de color LUV
 * (2) Magnitud del gradiente
 * (3) Canales de gradiente cuantificados.
 * 
 * @param src: Imagen de la cual se quieren calcular las características.
 * @param shrink: Cantidad para submuestrear los canales calculados
 * @return std::vector<cv::Mat>: Vector de cv::Mat con las imágenes correspondientes a las distintas
 *                               características.
 *
 */
std::vector<cv::Mat> channelsCompute(cv::Mat src, std::string colorSpace, int shrink){

  productChnsCompute productCompute;

  int smooth = 1;
  ChannelsLUVExtractor channExtract{false, smooth};
  GradMagExtractor gradMagExtract{5};
  GradHistExtractor gradHistExtract{2,6,1,0}; //{4,6,1,1}; // <--- JM: Cuidado!! Estos parámetros dependerán del clasificador entrenado?

  int dChan = src.channels();
  int h = src.size().height;
  int w = src.size().width;

  int crop_h = h % shrink;
  int crop_w = w % shrink;

  h = h - crop_h;
  w = w - crop_w;
  
  cv::Rect cropImage = cv::Rect(0,0,w, h);
  cv::Mat imageCropped = src(cropImage);

  //printf("%d %d\n",h,w );
  cv::Mat luv_image;
  std::vector<cv::Mat> luvImage;
  if(colorSpace != "LUV"){
    luvImage = channExtract.extractFeatures(imageCropped); //IMAGENES ESCALA DE GRISES??
    /*split(imageCropped, luvImage);
    cv::imshow("",luvImage[0]);*/
    cv::Mat luvImageChng;
    merge(luvImage, luv_image);
  }else{
    luv_image = imageCropped;
    split(luv_image, luvImage);
  }

  luv_image = convTri(luv_image, smooth);

  std::vector<cv::Mat> gMagOrient = gradMagExtract.extractFeatures(luv_image);

  //-------------------------------------------------------------
  /*cv::Mat img3;
  gMagOrient[0].convertTo(img3, CV_32F);
  float *valueM = img3.ptr<float>();

  printf("M: \n");
  for(int i = 0; i < 15; i++)
    printf("%.4f ", valueM[i] );
  printf("\n");*/
  //printf("----------------------------\n");

  std::vector<cv::Mat> gMagHist = gradHistExtract.extractFeatures(luv_image, gMagOrient);
  //printf("---------------------------- 2\n");

  std::vector<cv::Mat> chnsCompute;
  for(int i = 0; i < 3/*luvImage.size()*/; i++){   //FALTA HACER RESAMPLE TAMAÑO/SHRINK PARA RETORNAR EL RESULTADO COMO ADDCHNS
    cv::Mat resampleLuv = ImgResample(luvImage[i], w/shrink, h/shrink);
    chnsCompute.push_back(resampleLuv);
  }

  cv::Mat resampleMag = ImgResample(gMagOrient[0], w/shrink, h/shrink);
  chnsCompute.push_back(resampleMag);

  for(int i = 0; i < gMagHist.size(); i++){
    cv::Mat resampleHist = ImgResample(gMagHist[i], w/shrink, h/shrink);
    chnsCompute.push_back(resampleHist);
  }

  return chnsCompute;
}




