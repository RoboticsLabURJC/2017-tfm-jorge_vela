/** ------------------------------------------------------------------------
 *
 *  @brief Test of Image Resample
 *  @author Jorge Vela Peña
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2019/07/08
 *
 *  ------------------------------------------------------------------------ */


#include <channels/Utils.h>

#include "gtest/gtest.h"
#include <opencv/cv.hpp>

#include <iostream>


using namespace cv;
using namespace std;


class TestUtils: public testing::Test
{
public:
  Utils utils;
  virtual void SetUp()
    {
    }

  virtual void TearDown()
    {
    }
};

TEST_F(TestUtils, TestResampleGrayImage){
  cv::Mat image = cv::imread("images/imgGrayScale.jpeg", cv::IMREAD_GRAYSCALE); 
  cv::Mat image2 = cv::imread("images/index3.jpeg", cv::IMREAD_GRAYSCALE); 

  cv::Mat imageMatlab = cv::imread("images/mask_image_gray.jpeg", cv::IMREAD_GRAYSCALE); 

  cv:Mat dst = utils.ImgResample(image2, 35,29);  

  cv::Mat diff = imageMatlab - dst;

  cv::Mat img1;
  diff.convertTo(img1, CV_32F);    
  float *diffImageVals = diff.ptr<float>();


  int valTot = 0;
  int diffTot = 0;
  for(int i= 0; i < dst.size().height*dst.size().width; i++)
  {
      if(diffImageVals[i] > 15)
      {
        diffTot = diffTot + diffImageVals[i];
        valTot = valTot + 1; 
      }
  }
  //printf("DifTot --> %d\n", valTot);
  //ASSERT_TRUE(valTot < 40);
}


TEST_F(TestUtils, TestResampleColorImage)
{
  cv::Mat imageMatlab = cv::imread("images/mask_image.jpg", cv::IMREAD_COLOR); 
  cv::Mat image = cv::imread("images/index.jpeg", cv::IMREAD_COLOR); 
  int h = 0;
  int w = 0;
  if(image.size().height % 2 == 0)
  {
    h = image.size().height / 2;
  }
  else
  {
    h = image.size().height / 2 + 1;
  }

  if(image.size().height % 2 == 0)
  {
    w = image.size().width / 2;
  }
  else
  {
    w = image.size().width / 2 + 1;
  }

  cv:Mat dst = utils.ImgResample(image, w, h);

  transpose(dst, dst);

  cv::Mat bgr_dst[3];
  split(dst,bgr_dst);

  cv::Mat bgr_resample[3];
  split(imageMatlab,bgr_resample);
  
  FileStorage fs1;
  FileStorage fs2;
  FileStorage fs3;
  fs1.open("yaml/imresample_1.yaml", FileStorage::READ);
  fs2.open("yaml/imresample_2.yaml", FileStorage::READ);
  fs3.open("yaml/imresample_3.yaml", FileStorage::READ);

  FileNode rows = fs1["res1"]["rows"];
  FileNode cols = fs1["res1"]["cols"];
  
  FileNode imReample1 = fs1["res1"]["data"];
  FileNode imReample2 = fs2["res2"]["data"];
  FileNode imReample3 = fs3["res3"]["data"];

  cv::Mat bgr_dst0_chng;
  bgr_dst[0].convertTo(bgr_dst0_chng, CV_32F);
  //transpose(bgr_dst0_chng, bgr_dst0_chng);
  float *data = bgr_dst0_chng.ptr<float>();

  int difPixels = 0;
  for(int j = 0; j < (int)cols; j++)
  {
    for(int i = 0; i < (int)rows; i++)
    {
      if(abs(data[i + j] -  (float)imReample3[i + j])> 10)
      {
        difPixels = difPixels + 1;
      }
    }
  }

  ASSERT_TRUE(difPixels < 350);

  cv::Mat bgr_dst1_chng;
  bgr_dst[1].convertTo(bgr_dst1_chng, CV_32F);
  float *data2 = bgr_dst1_chng.ptr<float>();

  int difPixels2 = 0;
  for(int j = 0; j < (int)cols; j++)
  {
    for(int i = 0; i < (int)rows; i++)  
    {
      if(abs(data2[i + j] -  (float)imReample2[i + j])> 10)
      {
        difPixels2 = difPixels2 + 1;
      }
    }
  }

  ASSERT_TRUE(difPixels2 < 350);

  cv::Mat bgr_dst2_chng;
  bgr_dst[2].convertTo(bgr_dst2_chng, CV_32F);
  float *data3 = bgr_dst2_chng.ptr<float>();

  int difPixels3 = 0;
  for(int j = 0; j < (int)cols; j++)
  {
    for(int i = 0; i < (int)rows; i++)
    {
      if(abs(data3[i + j] -  (float)imReample1[i + j])> 10)
      {
        difPixels3 = difPixels3 + 1;
      }
    }
  }
  ASSERT_TRUE(difPixels3 < 350);


  int difPixels4 = 0;
  for(int j = 0; j < (int)cols; j++)
  {
    for(int i = 0; i < (int)rows; i++)
    {
      if(abs(data3[i + j] -  (float)imReample1[i + j])> 10 or abs(data2[i + j] -  (float)imReample2[i + j])> 10 or abs(data[i + j] -  (float)imReample3[i + j])> 10) 
      {
        difPixels4 = difPixels4 + 1;
      }
    }
  }
  ASSERT_TRUE(difPixels4 < 350);
}

TEST_F(TestUtils, TestResampleConv)
{
  cv::Mat image = cv::imread("images/index3.jpeg", cv::IMREAD_GRAYSCALE);
  cv::Mat imgConv = utils.convTri(image, 5);

  transpose(imgConv, imgConv);

  cv::Mat img1;
  imgConv.convertTo(img1, CV_32F);    
  float *valuesImgConv = img1.ptr<float>();

  FileStorage fs1;
  fs1.open("yaml/convTri.yml", FileStorage::READ);

  FileNode rows = fs1["J"]["rows"];
  FileNode cols = fs1["J"]["cols"];
  FileNode imgMatlab = fs1["J"]["data"];

  for(int i=0;i<(int)rows*(int)cols;i++)
  { 
    ASSERT_TRUE(abs((int)valuesImgConv[i] - (int)imgMatlab[i]) < 1.1);
  }
}


TEST_F(TestUtils, TestChannelsCompute)
{
  cv::Mat image = cv::imread("images/index3.jpeg", cv::IMREAD_COLOR);
  std::vector<cv::Mat> pChnsCompute;
  pChnsCompute = utils.channelsCompute(image, 4);

  cv::Mat testMag;
  transpose(pChnsCompute[3], testMag);

  cv::Mat imgMag;
  testMag.convertTo(imgMag, CV_32F);    
  float *valuesImgMag = imgMag.ptr<float>();

  //printf("%f\n", valuesImgMag[1] );
  FileStorage fs1;
  fs1.open("yaml/TestMagChnsCompute.yml", FileStorage::READ);
  FileNode rows = fs1["M"]["rows"];
  FileNode cols = fs1["M"]["cols"];
  FileNode imgMagMatlab = fs1["M"]["data"];

  //for(int i=0;i<14*17 /*(int)rows*(int)cols*/;i++)
  //{ 
    //printf("%.4f %.4f \n", (float)valuesImgMag[i], (float)imgMagMatlab[i] );
    //ASSERT_TRUE(abs((float)valuesImgMag[i] - (float)imgMagMatlab[i]) < 1.e-2f);
  //}

}

/*
TEST_F(TestUtils, TestGetScales)
{
  int nPerOct = 8;
  int nOctUp = 1;
  int shrink = 4;
  int size[2] = {19,22};
  int minDS[2] = {16,16};
  std::vector<float> scales = utils.getScales(nPerOct, nOctUp, minDS, shrink, size);
  std::vector<float> check = {2.1463, 1.8537, 1.6589, 1.4632, 1.2684, 1.0737, 0.8779};

  for(int i = 0; i < scales.size(); i++){
    ASSERT_TRUE(abs(scales[i]-check[i])<1.e-3f);
  }
}
*/

/*

TEST_F(TestUtils, TestGetScalesChangeVals)
{
  int nPerOct = 7;
  int nOctUp = 0;
  int shrink = 4;
  int size[2] = {30,30};
  int minDS[2] = {16,16};
  std::vector<float> scales = utils.getScales(nPerOct, nOctUp, minDS, shrink, size);
  std::vector<float> check = {1.0667, 0.9333, 0.8000, 0.6667, 0.5333};

  for(int i = 0; i < scales.size(); i++){
    ASSERT_TRUE(abs(scales[i]-check[i])<1.e-3f);
  }
}
*/
/*
TEST_F(TestUtils, chnsPyramids)
{
  cv::Mat image = cv::imread("images/index.jpeg", cv::IMREAD_COLOR);
  std::vector<cv::Mat> pyramid = utils.chnsPyramids(image);
  //printf("%d\n", pyramid[0].channels());
}
*/

/*
TEST_F(TestUtils, TestChannelsPyramids)
{
  cv::Mat image = cv::imread("images/index.jpeg", cv::IMREAD_COLOR);

  FileStorage pPyramid;
  pPyramid.open("yaml/pPyramid.yml", FileStorage::READ);
  int nPerOct = pPyramid["nPerOct"]["data"][0];
  int nOctUp = pPyramid["nOctUp"]["data"][0];
  int nApprox = pPyramid["nApprox"]["data"][0];
  int pad[2] = {pPyramid["pad"]["data"][0], pPyramid["pad"]["data"][1]};
  int shrink = pPyramid["pChns.shrink"]["data"];

  int sz[2] = {image.size().height, image.size().width};
  std::vector<int> minDS = {48, 84};


  ASSERT_TRUE(nPerOct == 10);
  ASSERT_TRUE(nOctUp == 1);
  ASSERT_TRUE(shrink == 2);
  ASSERT_TRUE(nApprox == 9);

  //LLAMADA A CHNSPYRAMIDS CON LA IMAGEN, RECIBIENDO LA PIRAMIDE COMPLETA
  std::vector<cv::Mat> pyramid = utils.chnsPyramids(image, nOctUp, nPerOct, nApprox, shrink, minDS);
  printf("%d\n",pyramid.size());

  //CARGAR EL FILTRO CREADO POR MATLAB DESDE UN YML
  FileStorage filter;
  filter.open("yaml/filterTest.yml", FileStorage::READ);

  //OBTENER EL NOMBRE DE LOS DISTINTOS FILTROS PARA ESTE CASO
  std::vector<std::string> namesFilters;
  for(int i = 1; i < 5; i++){
    for(int j = 1; j< 11; j++){
      std::string name  = "filter_" + to_string(j) + "_" + to_string(i);
      namesFilters.push_back(name);
    }
  }

  //SE CARGAN LOS DISTINTOS FILTROS, CON LOS NOMBRES ANTERIORES DESDE EL YML
  std::vector<cv::Mat> filters;
  for(int k = 0; k < namesFilters.size(); k++){
    FileNode filterData = filter[namesFilters[k].c_str()]["data"];
    FileNode filterRows = filter[namesFilters[k].c_str()]["rows"];
    FileNode filterCols = filter[namesFilters[k].c_str()]["cols"];

    float* filt = new float[25*sizeof(float)];

    for(int i = 0; i < (int)filterRows; i++){
      for(int j = 0; j < (int)filterCols; j++){
        float x = (float)filterData[i*5+j];
        filt[i*5+j] = x;
      }
    }

    cv::Mat filterConver = cv::Mat(5,5, CV_32F, filt);
    transpose(filterConver,filterConver);
    float *O = filterConver.ptr<float>();

    filters.push_back(filterConver);//(filt);
  }

  //EJEMPLO PARA UNA ESCALA, QUE TIENE 10 CANALES
  cv::Mat bgr_dst[10];
  split(pyramid[2],bgr_dst);

  //SE REPITE UNA ESCALA PARA PASAR POR LOS FILTROS
  cv::Mat G;
  pyramid[0].copyTo(G);
  std::vector<cv::Mat> C_repMat;
  for(int i = 0; i < 10; i++){
    pyramid[0].copyTo(G);
    C_repMat.push_back(G);
  }


  //SE CONVOLUCIONA UNA IMAGEN CON LOS FILTROS Y SE OBTIENEN LAS IMAGENES DE SALIDA
  std::vector<cv::Mat> out_images;
  for(int j = 0; j < 4; j++){
    cv::Mat splitted[10];
    split(C_repMat[j],splitted);
    for(int i = 0; i < 10; i++){
      cv::Mat out_image; 
      filter2D(splitted[i], out_image, -1 , filters[i+(10*j)], cv::Point( -1, -1 ), 0, cv::BORDER_REFLECT );
      out_images.push_back(out_image);
    }
  }
} */