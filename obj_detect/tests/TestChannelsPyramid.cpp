
#include <channels/ChannelsPyramid.h> 
#include "gtest/gtest.h"
#include <opencv/cv.hpp>

#include <iostream>


class TestChannelsPyramid: public testing::Test
{
public:
  ChannelsPyramid chnsPyramid;
  virtual void SetUp()
    {
    }

  virtual void TearDown()
    {
    }



};

TEST_F(TestChannelsPyramid, TestGetScales)
{
  int nPerOct = 8;
  int nOctUp = 1;
  int shrink = 4;
  int size[2] = {19,22};
  int minDS[2] = {16,16};
  std::vector<float> scales = chnsPyramid.getScales(nPerOct, nOctUp, minDS, shrink, size);
  std::vector<float> check = {2.1463, 1.8537, 1.6589, 1.4632, 1.2684, 1.0737, 0.8779};

  for(int i = 0; i < scales.size(); i++){
    ASSERT_TRUE(abs(scales[i]-check[i])<1.e-3f);
  }
}


TEST_F(TestChannelsPyramid, channelsPyramid){
  cv::Mat image = cv::imread("images/index.jpeg", cv::IMREAD_COLOR);

  std::string nameOpts = "yaml/pPyramid.yml";
  bool loadOk = chnsPyramid.load(nameOpts.c_str());
  ASSERT_TRUE(loadOk);

  std::vector<cv::Mat> pyramid = chnsPyramid.getPyramid(image,1,10,9,4);

  ASSERT_TRUE(pyramid.size()==28);

  //CARGAR EL FILTRO CREADO POR MATLAB DESDE UN YML
  cv::FileStorage filter;
  filter.open("yaml/filterTest.yml", cv::FileStorage::READ);

  //OBTENER EL NOMBRE DE LOS DISTINTOS FILTROS PARA ESTE CASO
  std::vector<std::string> namesFilters;
  for(int i = 1; i < 5; i++){
    for(int j = 1; j< 11; j++){
      std::string name  = "filter_" + std::to_string(j) + "_" + std::to_string(i);
      namesFilters.push_back(name);
    }
  }

  //SE CARGAN LOS DISTINTOS FILTROS, CON LOS NOMBRES ANTERIORES DESDE EL YML
  std::vector<cv::Mat> filters;
  for(int k = 0; k < namesFilters.size(); k++){
    cv::FileNode filterData = filter[namesFilters[k].c_str()]["data"];
    cv::FileNode filterRows = filter[namesFilters[k].c_str()]["rows"];
    cv::FileNode filterCols = filter[namesFilters[k].c_str()]["cols"];

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


}
