/** ------------------------------------------------------------------------
 *
 *  @brief Test of Image Resample
 *  @author Jorge Vela Peña
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2020/09/30
 *
 *  ------------------------------------------------------------------------ */


#include <iostream>
#include <detectors/BadacostDetector.h>
#include <channels/ChannelsExtractorACF.h>
#include <channels/ChannelsExtractorLDCF.h>
#include <channels/Utils.h>
#include <opencv2/opencv.hpp>
#include "gtest/gtest.h"


#include <iostream>


class TestChannelsExtractorLDCF: public testing::Test
{
public:
  virtual void SetUp()
    {
    }

  virtual void TearDown()
    {
    }
};


TEST_F(TestChannelsExtractorLDCF, TestChannelsExtractorEstractFeaturesFromLDCF)
{
  cv::Mat image = cv::imread("images/coche_solo1.png", cv::IMREAD_COLOR);
  int shrink = 4;
  cv::Size padding;
  padding.width = 2;
  padding.height = 2;

  cv::FileStorage filter;
  filter.open("yaml/filterTest.yml", cv::FileStorage::READ);
  //OBTENER EL NOMBRE DE LOS DISTINTOS FILTROS PARA ESTE CASO
  std::vector<std::string> namesFilters;
  int num_filters_per_channel = 4;
  int num_channels = 10; //
  for(int i = 1; i <= num_filters_per_channel; i++)
  {
    for(int j = 1; j <= num_channels; j++)
    {
      std::string name  = "filter_" + std::to_string(j) + "_" + std::to_string(i);
      namesFilters.push_back(name);
    }
  }

  std::vector<cv::Mat> filters;
  std::vector<float> p;
  for(uint k = 0; k < namesFilters.size(); k++)
  {
    cv::FileNode filterData = filter[namesFilters[k]]["data"];
    cv::FileNode filterRows = filter[namesFilters[k]]["cols"];
    cv::FileNode filterCols = filter[namesFilters[k]]["rows"];

    cv::Mat filterConver = cv::Mat::zeros(filterRows, filterCols, CV_32F);
    p.clear();
    filterData >> p;
    memcpy(filterConver.data, p.data(), p.size()*sizeof(float));
    transpose(filterConver,filterConver);
    // NOTE: filter2D is a correlation and to do convolution as in Matlab's conv2 we have to flip the kernels in advance.
    //       We have do it when loading them from file.
    cv::flip(filterConver, filterConver, -1);
    filters.push_back(filterConver);
  }
  ChannelsExtractorLDCF ldcfExtractor(filters, padding, shrink,5,0.005,2,6,1,0);
  std::vector<cv::Mat> ldcfExtractFeatures = ldcfExtractor.extractFeatures(image);

  cv::Mat valFiltered;
  ldcfExtractFeatures[0].convertTo(valFiltered, CV_32F);
  float *ldcfExtracted = valFiltered.ptr<float>();

  cv::FileStorage fs1;
  bool file_exists = fs1.open("yaml/LDCF.yml", cv::FileStorage::READ);
  cv::FileNode rows = fs1["filtered_1"]["rows"];
  cv::FileNode cols = fs1["filtered_1"]["cols"];
  cv::FileNode Filtered = fs1["filtered_1"]["data"];

  int j = 0;
  int difference = 0;
  for(int y=0;y<(int)rows;y++)
  {
    for(int x=0;x<(int)cols;x++)
    {
      if(abs(ldcfExtracted[x*(int)cols+y] - (float)Filtered[j]) > 0.5){
        difference +=1;
      }
      j++;
    }
  }
  ASSERT_TRUE(difference < 40);
}



TEST_F(TestChannelsExtractorLDCF, TestChannelsExtractorEstractFeaturesFromLDCFSintetic)
{
//  float Isintetic[100] = {1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10};
//  cv::Mat sintetic = cv::Mat(10,10, CV_32FC1, Isintetic);
//  //copyMakeBorder( sintetic, sintetic, 3,3,2,2, cv::BORDER_CONSTANT, 0 );

//  std::vector<cv::Mat> completeImage;
//  int i = 0;
//  while(i < 10){
//    completeImage.push_back(sintetic);
//    i = i + 1;
//  }
 
//  //CARGAR EL FILTRO CREADO POR MATLAB DESDE UN YML
//  cv::FileStorage filter;
//  filter.open("yaml/filterTest.yml", cv::FileStorage::READ);

//  //OBTENER EL NOMBRE DE LOS DISTINTOS FILTROS PARA ESTE CASO
//  std::vector<std::string> namesFilters;
//  int num_filters_per_channel = 4;
//  int num_channels = 10; //
//  for(int i = 1; i <= num_filters_per_channel; i++)
//  {
//    for(int j = 1; j <= num_channels; j++)
//    {
//      std::string name  = "filter_" + std::to_string(j) + "_" + std::to_string(i);
//      namesFilters.push_back(name);
//    }
//  }

//  std::vector<cv::Mat> filters;
//  std::vector<float> p;
//  for(uint k = 0; k < namesFilters.size(); k++)
//  {
//    cv::FileNode filterData = filter[namesFilters[k]]["data"];
//    cv::FileNode filterRows = filter[namesFilters[k]]["cols"];
//    cv::FileNode filterCols = filter[namesFilters[k]]["rows"];

//    cv::Mat filterConver = cv::Mat::zeros(filterRows, filterCols, CV_32F);
//    p.clear();
//    filterData >> p;
//    memcpy(filterConver.data, p.data(), p.size()*sizeof(float));
//    transpose(filterConver,filterConver);

//    // NOTE: filter2D is a correlation and to do convolution as in Matlab's conv2 we have to flip the kernels in advance.
//    //       We have do it when loading them from file.
//    cv::flip(filterConver, filterConver, -1);
//    filters.push_back(filterConver);
//  }

//  //for(int kw = 0; kw < 40; kw++)
//  //std::cout << filters[kw] << std::endl;
//  int shrink = 2;
//  cv::Size padding;
//  padding.width = 3;
//  padding.height = 2;
//  ChannelsExtractorLDCF ldcfExtractor(filters, padding, shrink);
//  std::vector<cv::Mat> ldcfExtractImages = ldcfExtractor.extractFeaturesFromACF(completeImage);

//  cv::Mat valFiltered;
//  ldcfExtractImages[0].convertTo(valFiltered, CV_32F);
//  float *ldcfExtracted = valFiltered.ptr<float>();

//  cv::FileStorage fs1;
//  bool file_exists = fs1.open("yaml/LDFSintetic_1.yml", cv::FileStorage::READ);
//  cv::FileNode rows = fs1["filtered"]["rows"];
//  cv::FileNode cols = fs1["filtered"]["cols"];
//  cv::FileNode Filtered = fs1["filtered"]["data"];

//  int j = 0;
//  for(int y=0;y<(int)rows;y++)
//  {
//    for(int x=0;x<(int)cols;x++)
//    {
//      ASSERT_TRUE(abs(ldcfExtracted[x*(int)cols+y] - (float)Filtered[j]) < 0.2);
//      j++;
//    }
//  }

//  ldcfExtractImages[34].convertTo(valFiltered, CV_32F);
//  ldcfExtracted = valFiltered.ptr<float>();

//  file_exists = fs1.open("yaml/LDFSintetic_35.yml", cv::FileStorage::READ);
//  rows = fs1["filtered"]["rows"];
//  cols = fs1["filtered"]["cols"];
//  Filtered = fs1["filtered"]["data"];

//  j = 0;
//  for(int y=0;y<(int)rows;y++)
//  {
//    for(int x=0;x<(int)cols;x++)
//    {
//      ASSERT_TRUE(abs(ldcfExtracted[x*(int)cols+y] - (float)Filtered[j]) < 0.001);
//      j++;
//    }
//  }

//  ldcfExtractImages[17].convertTo(valFiltered, CV_32F);
//  ldcfExtracted = valFiltered.ptr<float>();

//  file_exists = fs1.open("yaml/LDFSintetic_18.yml", cv::FileStorage::READ);
//  rows = fs1["filtered"]["rows"];
//  cols = fs1["filtered"]["cols"];
//  Filtered = fs1["filtered"]["data"];

//  j = 0;
//  for(int y=0;y<(int)rows;y++)
//  {
//    for(int x=0;x<(int)cols;x++)
//    {
//      ASSERT_TRUE(abs(ldcfExtracted[x*(int)cols+y] - (float)Filtered[j]) < 0.001);
//      j++;
//    }
//  }

}





TEST_F(TestChannelsExtractorLDCF, TestChannelsExtractorEstractFeaturesFromLDCFSintetic2)
{

//  float Isintetic[25] = {1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5};
//  cv::Mat sintetic = cv::Mat(5,5, CV_32FC1, Isintetic);
//  //copyMakeBorder( sintetic, sintetic, 3,3,2,2, cv::BORDER_CONSTANT, 0 );

//  std::vector<cv::Mat> completeImage;
//  int i = 0;
//  while(i < 10){
//    completeImage.push_back(sintetic);
//    i = i + 1;
//  }
//  //CARGAR EL FILTRO CREADO POR MATLAB DESDE UN YML
//  cv::FileStorage filter;
//  filter.open("yaml/filterTest.yml", cv::FileStorage::READ);

//  //OBTENER EL NOMBRE DE LOS DISTINTOS FILTROS PARA ESTE CASO
//  std::vector<std::string> namesFilters;
//  int num_filters_per_channel = 4;
//  int num_channels = 10; //
//  for(int i = 1; i <= num_filters_per_channel; i++)
//  {
//    for(int j = 1; j <= num_channels; j++)
//    {
//      std::string name  = "filter_" + std::to_string(j) + "_" + std::to_string(i);
//      namesFilters.push_back(name);
//    }
//  }

//  std::vector<cv::Mat> filters;
//  std::vector<float> p;
//  for(uint k = 0; k < namesFilters.size(); k++)
//  {
//    cv::FileNode filterData = filter[namesFilters[k]]["data"];
//    cv::FileNode filterRows = filter[namesFilters[k]]["cols"];
//    cv::FileNode filterCols = filter[namesFilters[k]]["rows"];

//    cv::Mat filterConver = cv::Mat::zeros(filterRows, filterCols, CV_32F);
//    p.clear();
//    filterData >> p;
//    memcpy(filterConver.data, p.data(), p.size()*sizeof(float));
//    transpose(filterConver,filterConver);

//    // NOTE: filter2D is a correlation and to do convolution as in Matlab's conv2 we have to flip the kernels in advance.
//    //       We have do it when loading them from file.
//    cv::flip(filterConver, filterConver, -1);
//    filters.push_back(filterConver);
//  }

//  int shrink = 2;
//  cv::Size padding;
//  padding.width = 3;
//  padding.height = 2;
//  ChannelsExtractorLDCF ldcfExtractor(filters, padding, shrink);
//  std::vector<cv::Mat> ldcfExtractImages = ldcfExtractor.extractFeaturesFromACF(completeImage);

//  cv::Mat valFiltered;
//  ldcfExtractImages[22].convertTo(valFiltered, CV_32F);
//  float *ldcfExtracted = valFiltered.ptr<float>();

//  cv::FileStorage fs1;
//  bool file_exists = fs1.open("yaml/LDFSintetic_size5_23.yml", cv::FileStorage::READ);
//  cv::FileNode rows = fs1["filtered"]["rows"];
//  cv::FileNode cols = fs1["filtered"]["cols"];
//  cv::FileNode Filtered = fs1["filtered"]["data"];

//  int j = 0;
//  for(int y=0;y<(int)rows;y++)
//  {
//    for(int x=0;x<(int)cols;x++)
//    {
//      ASSERT_TRUE(abs(ldcfExtracted[x*(int)cols+y] - (float)Filtered[j]) < 0.01);
//      j++;
//    }
//  }

//  ldcfExtractImages[35].convertTo(valFiltered, CV_32F);
//  ldcfExtracted = valFiltered.ptr<float>();

//  file_exists = fs1.open("yaml/LDFSintetic_size5_36.yml", cv::FileStorage::READ);
//  rows = fs1["filtered"]["rows"];
//  cols = fs1["filtered"]["cols"];
//  Filtered = fs1["filtered"]["data"];

//  j = 0;
//  for(int y=0;y<(int)rows;y++)
//  {
//    for(int x=0;x<(int)cols;x++)
//    {
//      printf("%f %f \n",ldcfExtracted[x*(int)cols+y], (float)Filtered[j] );
//      ASSERT_TRUE(abs(ldcfExtracted[x*(int)cols+y] - (float)Filtered[j]) < 0.001);
//      j++;
//    }
//  }


}

















