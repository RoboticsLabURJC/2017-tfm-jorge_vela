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
#include <cmath>
#include <functional>
#include <numeric>
#include <random>

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
void
ImgResample
  (
  cv::Mat src,
  cv::Mat& dst,
  int width,
  int height,
  std::string method,
  float norm
  )
{

  if (method == "antialiasing")
  {
    cv::resize(src, dst, cv::Size(width, height), 0, 0, cv::INTER_AREA);
  }
  else
  {
    cv::resize(src, dst, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
  }

  dst *= norm;
//  cv::Mat norm_mat(dst.size(), dst.type(), cv::Scalar(norm));
//  cv::multiply(dst, norm_mat, dst);
}


/**
 * Función Imgresample. Encargada de redimensionar una imagen de entrada, al tamaño de ancho y alto
 * que se le pase por parámetros.
 *
 * @param src: Imagen que se quiere redimensionar
 * @param dst: Imagen destino.
 * @param width: Ancho de la imagen de salida
 * @param height: Alto de la imagen de salida
 * @param norm: [1] Valor por el que se multiplican los píxeles de salida
 * @return cv::Mat: Imagen redimensionada
 *
 */
void
ImgResample
  (
  cv::UMat src,
  cv::UMat& dst,
  int width,
  int height,
  std::string method,
  float norm
  )
{
  if (method == "antialiasing")
  {
    cv::resize(src, dst, cv::Size(width, height), 0, 0, cv::INTER_AREA);
  }
  else
  {
    cv::resize(src, dst, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
  }

  cv::UMat norm_mat(dst.size(), dst.type(), cv::Scalar(norm));
  cv::multiply(dst, norm_mat, dst);
}

/**
 * Funcion convTri. Convoluciona una imagen por un filtro de triangulo 2D. 
 *
 * @param input_image: Imagen de entrada la cual se quiere convolucionar.
 * @param kernel_size: Tamaño del kernel (radio) que se quiere para el filtro.
 *
 * @return cv::Mat: Imagen de retorno despues del filtro.
 */
void
convTri
  (
  cv::Mat input_image,
  cv::Mat& dst,
  int kernel_size
  )
{
  cv::copyMakeBorder(input_image, dst, kernel_size, kernel_size, kernel_size, kernel_size, cv::BORDER_REFLECT, 0);

  // FUNCION CONVTRI
  int widthSize = input_image.size().width;
  int heightSize = input_image.size().height;
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
  kernel /= valReduce;

  cv::Mat aux;
  cv::filter2D(dst, aux, CV_32FC1 , kernel, cv::Point( -1, -1 ), 0, cv::BORDER_CONSTANT );
  cv::filter2D(aux, dst, CV_32FC1 , kernel.t(), cv::Point( -1, -1 ), 0, cv::BORDER_CONSTANT );
  cv::Rect myRoi2(kernel_size, kernel_size, widthSize, heightSize);
  dst = dst(myRoi2);
}

/**
 * Funcion convTri. Convoluciona una imagen por un filtro de triangulo 2D. 
 *
 * @param input_image: Imagen de entrada la cual se quiere convolucionar.
 * @param kernel_size: Tamaño del kernel (radio) que se quiere para el filtro.
 *
 * @return cv::Mat: Imagen de retorno despues del filtro.
 */
void
convTri
  (
  cv::UMat input_image,
  cv::UMat& dst,
  int kernel_size
  )
{

  int widthSize = input_image.size().width;
  int heightSize = input_image.size().height;
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
  kernel /= valReduce;

  cv::copyMakeBorder(input_image, dst, kernel_size, kernel_size, kernel_size, kernel_size, cv::BORDER_REFLECT, 0);

  cv::UMat aux;
  cv::filter2D(dst, aux, CV_32FC1 , kernel, cv::Point( -1, -1 ), 0, cv::BORDER_CONSTANT );
  cv::filter2D(aux, dst, CV_32FC1 , kernel.t(), cv::Point( -1, -1 ), 0, cv::BORDER_CONSTANT );
  cv::Rect myRoi2(kernel_size, kernel_size, widthSize, heightSize);
  dst = dst(myRoi2);
}












std::vector<int>
create_random_indices
  (
  int n
  )
{
  std::vector<int> indices(n);
  std::iota(indices.begin(), indices.end(), 0);
  std::shuffle(indices.begin(), indices.end(), std::mt19937(std::random_device()()));
  return indices;
}

float
readScalarFromFileNode
  (
  cv::FileNode fn
  )
{
  cv::FileNode data = fn["data"];
  std::vector<float> p;
  data >> p;

  return static_cast<float>(p[0]);
}


cv::Mat
readMatrixFromFileNode
  (
  cv::FileNode fn
  )
{
  int rows = static_cast<int>(fn["rows"]);
  int cols = static_cast<int>(fn["cols"]);
  cv::Mat matrix = cv::Mat::zeros(rows, cols, CV_32F);
  cv::FileNode data = fn["data"];
  std::vector<float> p;
  data >> p;
  memcpy(matrix.data, p.data(), p.size()*sizeof(float));

  return matrix;
}

cv::Mat
readMatrixFromFileNodeWrongBufferMatlab
  (
  cv::FileNode fn
  )
{
  int rows = static_cast<int>(fn["cols"]);
  int cols = static_cast<int>(fn["rows"]);
  cv::Mat matrix = cv::Mat::zeros(rows, cols, CV_32F);
  cv::FileNode data = fn["data"];
  std::vector<float> p;
  data >> p;
  memcpy(matrix.data, p.data(), p.size()*sizeof(float));

  return matrix;
}
