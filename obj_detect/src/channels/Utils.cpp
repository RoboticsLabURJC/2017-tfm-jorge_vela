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
cv::Mat ImgResample(cv::Mat src, int width, int height, int norm)
{
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
cv::Mat convTri(cv::Mat input_image, int kernel_size)
{
  float valReduce = (kernel_size + 1)*(kernel_size + 1);
  cv::Mat kernel = cv::Mat::zeros(1, kernel_size*2+1, CV_32FC1);
    
  for(int i = 1; i <= kernel_size + 1; i++)
  {
    kernel.at<float>(0, i-1) = static_cast<float>(i);
  }

  for(int j = kernel_size + 1; j < kernel.size().width ; j++)
  {
    kernel.at<float>(0, j) = kernel.at<float>(0, j-1)-1;
  }

  //std::cout << "kernel=" << kernel << std::endl;
  kernel /= valReduce;

  cv::Mat output_image;
  cv::Mat help_image;
  filter2D(input_image, help_image, CV_32FC1 , kernel, cv::Point( -1, -1 ), 0, cv::BORDER_CONSTANT );
  cv::Mat kernel_t;
  transpose(kernel, kernel_t);
  filter2D(help_image, output_image, CV_32FC1 , kernel_t, cv::Point( -1, -1 ), 0, cv::BORDER_CONSTANT );

  return output_image;
}

