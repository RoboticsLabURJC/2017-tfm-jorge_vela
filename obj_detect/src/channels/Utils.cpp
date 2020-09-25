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

  //cv::Scalar value = cv::Scalar( rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255) );
  //copyMakeBorder( input_image, input_image, kernel_size, kernel_size, kernel_size, kernel_size, cv::BORDER_CONSTANT, 0 );




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

