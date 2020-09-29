/** ------------------------------------------------------------------------
 *
 *  @brief Test of Image Resample
 *  @author Jorge Vela Peña
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2019/07/08
 *
 *  ------------------------------------------------------------------------ */

#include <channels/Utils.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "gtest/gtest.h"

using namespace cv;
using namespace std;

class TestUtils: public testing::Test
{
public:
  virtual void SetUp()
    {
    }

  virtual void TearDown()
    {
    }
};

/*TEST_F(TestUtils, TestResampleGrayImage){
  cv::Mat image = cv::imread("images/imgGrayScale.jpeg", cv::IMREAD_GRAYSCALE); 
  cv::Mat image2 = cv::imread("images/index3.jpeg", cv::IMREAD_GRAYSCALE); 

  cv::Mat imageMatlab = cv::imread("images/mask_image_gray.jpeg", cv::IMREAD_GRAYSCALE); 

  cv::Mat dst = ImgResample(image2, 35,29);

  cv::Mat diff = imageMatlab - dst;

  cv::Mat img1;
  diff.convertTo(img1, CV_32F);    
  float *diffImageVals = diff.ptr<float>();


  int valTot = 0;
  int diffTot = 0;
  for(int i= 0; i < dst.size().height*dst.size().width; i++)
  {
      if(diffImageVals[i] > 1)
      {
        diffTot = diffTot + diffImageVals[i];
        valTot = valTot + 1; 
      }
  }
  ASSERT_TRUE(valTot < 10);
}*/

TEST_F(TestUtils, TestResample1){
  float I[9] = {1 ,1 ,1,2,2,2,3,3,3};
  cv::Mat dummy_query = cv::Mat(3,3, CV_32F, I);
  cv::Mat dst = ImgResample(dummy_query, 4,4, "linear");

  transpose(dst, dst);
  cv::Mat img1;
  dst.convertTo(img1, CV_32F);    
  float *valuesImgRes = img1.ptr<float>();


  FileStorage fs1;  
  bool file_exists = fs1.open("yaml/TestImresample1.yml", FileStorage::READ);
  ASSERT_TRUE(file_exists);
  FileNode rows = fs1["resample"]["rows"];
  FileNode cols = fs1["resample"]["cols"];
  FileNode imgMatlab = fs1["resample"]["data"];

  for(int i=0;i<(int)rows*(int)cols;i++)
  { 
    ASSERT_TRUE(abs((float)valuesImgRes[i] - (float)imgMatlab[i]) < 0.01);
  }
}


TEST_F(TestUtils, TestResample2){
  float I[9] = {1 ,1 ,1,2,2,2,3,3,3};
  cv::Mat dummy_query = cv::Mat(3,3, CV_32F, I);
  cv::Mat dst = ImgResample(dummy_query, 4,4,"linear",2);

  transpose(dst, dst);
  cv::Mat img1;
  dst.convertTo(img1, CV_32F);    
  float *valuesImgRes = img1.ptr<float>();

  FileStorage fs1;
  bool file_exists = fs1.open("yaml/TestImresample2.yml", FileStorage::READ);
  ASSERT_TRUE(file_exists);


  FileNode rows = fs1["resample"]["rows"];
  FileNode cols = fs1["resample"]["cols"];
  FileNode imgMatlab = fs1["resample"]["data"];
  
  for(int i=0;i<(int)rows*(int)cols;i++)
  { 
    ASSERT_TRUE(abs((float)valuesImgRes[i] - (float)imgMatlab[i]) < 0.01);
  }
}


TEST_F(TestUtils, TestResampleReduce){
  float I[42] = {1 ,1 ,1,2,2,2,2, 2,2,2,3,3,3,3,3,3,3,4,4,4,4,4,4,4,5,5,5,5,5,5,5,6,6,6,6,6,6,6,7,7,7,7};
  cv::Mat dummy_query = cv::Mat(6,7, CV_32F, I);
  cv::Mat dst = ImgResample(dummy_query, 3,4,"antialiasing",4);

  transpose(dst, dst);
  cv::Mat img1;
  dst.convertTo(img1, CV_32F);    
  float *valuesImgRes = img1.ptr<float>();
  

  FileStorage fs1;  
  bool file_exists = fs1.open("yaml/TestImresampleReduce.yml", FileStorage::READ);
  ASSERT_TRUE(file_exists);

  FileNode rows = fs1["resample"]["rows"];
  FileNode cols = fs1["resample"]["cols"];
  FileNode imgMatlab = fs1["resample"]["data"];

  for(int i=0;i<(int)rows*(int)cols;i++)
  { 
    ASSERT_TRUE(abs((float)valuesImgRes[i] - (float)imgMatlab[i]) < 0.01);
  }
}

TEST_F(TestUtils, TestResampleReal){
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

  cv::Mat img2;
  image.convertTo(img2, CV_32F, 1.0 / 255, 0);

  cv::Mat dst = ImgResample(img2, 10, 10, "antialiasing", 2);

  std::vector<cv::Mat> splitted;
  split(dst, splitted);
 
  cv::Mat dst1;
  transpose(splitted[0], dst1);
  cv::Mat img1;
  dst1.convertTo(img1, CV_32F);    
  float *valuesImgRes = img1.ptr<float>();
  
  FileStorage fs1;
  fs1.open("yaml/TestImresampleReal_1.yml", FileStorage::READ);
  FileNode rows = fs1["resample"]["rows"];
  FileNode cols = fs1["resample"]["cols"];
  FileNode imgMatlab = fs1["resample"]["data"];

  for(int i=0;i<(int)rows*(int)cols;i++)
  { 
    ASSERT_TRUE(abs((float)valuesImgRes[i] - (float)imgMatlab[i]) < 0.1);
  }


  transpose(splitted[1], dst1);    
  dst1.convertTo(img1, CV_32F); 
  float *valuesImgRes1 = img1.ptr<float>();
  
  fs1.open("yaml/TestImresampleReal_2.yml", FileStorage::READ);
  rows = fs1["resample"]["rows"];
  cols = fs1["resample"]["cols"];
  imgMatlab = fs1["resample"]["data"];

  for(int i=0;i<(int)rows*(int)cols;i++)
  { 
    ASSERT_TRUE(abs((float)valuesImgRes1[i] - (float)imgMatlab[i]) < 0.1);
  }

  transpose(splitted[2], dst1);
  dst1.convertTo(img1, CV_32F);    
  float *valuesImgRes2 = img1.ptr<float>();
  
  fs1.open("yaml/TestImresampleReal_3.yml", FileStorage::READ);
  rows = fs1["resample"]["rows"];
  cols = fs1["resample"]["cols"];
  imgMatlab = fs1["resample"]["data"];

  for(int i=0;i<(int)rows*(int)cols;i++)
  { 
    ASSERT_TRUE(abs((float)valuesImgRes2[i] - (float)imgMatlab[i]) < 0.1);
  }

}

void testResample
  (
  std::string image_path,
  std::string matlab_result_yaml_file
  )
{
  cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
  cv::cvtColor(image, image, cv::COLOR_BGR2RGB); // In Matlab channels order is RGB unlike OpenCV.

  // Read the imResample resuls from Matlab.
  FileStorage fs1;
  bool file_exists = fs1.open(matlab_result_yaml_file, FileStorage::READ);
  ASSERT_TRUE(file_exists);

  FileNode scale_fn = fs1["scale"]["data"];
  vector<float> p;
  scale_fn >> p;
  float scale = p[0];

//  FileNode norm_fn = fs1["norm"]["data"];
//  p.clear();
//  norm_fn >> p;
//  float norm = p[0];

  FileNode method_fn = fs1["method"]["data"];
  p.clear();
  method_fn >> p;
  char p_str[15];
  for (int i=0; i<15; i++)
  {
    p_str[i] = static_cast<char>(p[i]);
  }
  std::string method(p_str);

  int w, h;
  if (method == "bilinear")
  {
    w = round(scale*image.size().width);
    h = round(scale*image.size().height);
  }
  else //if (method == "nearest")
  {
    w = ceil(scale*image.size().width);
    h = ceil(scale*image.size().height);
  }

  // Read matlab results size from yaml
  FileNode rows = fs1["img_resampled_1"]["rows"];
  FileNode cols = fs1["img_resampled_1"]["cols"];

  // Actual call to our implementation of resampling.
  cv::Mat resampledImage = ImgResample(image, w, h);

  std::vector<cv::Mat> imResample_matlab_float(3);
  std::vector<cv::Mat> resampledImage_channels(3);
  std::vector<cv::Mat> resampledImage_channels_float(3);
  for (int i=0; i<3; i++)
  {
    imResample_matlab_float[i] = cv::Mat::zeros(rows, cols, CV_32FC1);
    resampledImage_channels[i] = cv::Mat::zeros(h, w, CV_8UC1);
    resampledImage_channels_float[i] = cv::Mat::zeros(h, w, CV_32FC1);
  }

  FileNode imResample1 = fs1["img_resampled_1"]["data"];
  p.clear();
  imResample1 >> p;
  memcpy(imResample_matlab_float[0].data, p.data(), p.size()*sizeof(float));

  FileNode imResample2 = fs1["img_resampled_2"]["data"];
  p.clear();
  imResample2 >> p;
  memcpy(imResample_matlab_float[1].data, p.data(), p.size()*sizeof(float));

  FileNode imResample3 = fs1["img_resampled_3"]["data"];
  p.clear();
  imResample3 >> p;
  memcpy(imResample_matlab_float[2].data, p.data(), p.size()*sizeof(float));

  // Split the channels of the resampled image.
  split(resampledImage, resampledImage_channels);
  resampledImage_channels[0].convertTo(resampledImage_channels_float[0], CV_32FC1);
  resampledImage_channels[1].convertTo(resampledImage_channels_float[1], CV_32FC1);
  resampledImage_channels[2].convertTo(resampledImage_channels_float[2], CV_32FC1);

  // Compare pixels from resampledImage and resampledImageMatlab
//  std::cout << "resampledImage_channels_float[0].size()=" << resampledImage_channels_float[0].size() << std::endl;
//  std::cout << "resampledImage_channels_float[0].type()=" << resampledImage_channels_float[0].type() << std::endl;
//  std::cout << "imResample_matlab_float[0].size()=" << imResample_matlab_float[0].size() << std::endl;
//  std::cout << "imResample_matlab_float[0].type()=" << imResample_matlab_float[0].type() << std::endl;

#ifdef SHOW_IMAGES
  cv::imshow("image", image);
  cv::Mat kk;
  imResample_matlab_float[0].convertTo(kk, CV_8UC1);
  cv::imshow("matlab", kk);
  resampledImage_channels_float[0].convertTo(kk, CV_8UC1);
  cv::imshow("cpp", kk);
  cv::waitKey();
#endif

  cv::Mat absDiff0 = cv::abs(resampledImage_channels_float[0] - imResample_matlab_float[0]);
  cv::Mat absDiff1 = cv::abs(resampledImage_channels_float[1] - imResample_matlab_float[1]);
  cv::Mat absDiff2 = cv::abs(resampledImage_channels_float[2] - imResample_matlab_float[2]);

#ifdef SHOW_IMAGES
  absDiff0.convertTo(kk, CV_8UC1);
  cv::imshow("diff 0", kk);
  absDiff1.convertTo(kk, CV_8UC1);
  cv::imshow("diff 1", kk);
  absDiff2.convertTo(kk, CV_8UC1);
  cv::imshow("diff 2", kk);
  cv::waitKey();
#endif

  int num_differences = 0;
  for (int i = 0; i < h; i++)
  {
    for (int j = 0; j < w; j++)
    {
      if ((absDiff0.at<uint8_t>(i,j)>1) || (absDiff1.at<uint8_t>(i,j)>1) || (absDiff2.at<uint8_t>(i,j)>1))
      {
        num_differences++;
      }
    }
  }

//  std::cout << "num_differences=" << num_differences << std::endl;
  ASSERT_TRUE(num_differences < 50);
}


TEST_F(TestUtils, TestResampleColorImage038Bilinear)
{
  testResample("images/index.jpeg", "yaml/index_jpeg_imResample_scale_0_38_method_bilinear_norm_1.yaml");
}

TEST_F(TestUtils, TestResampleColorImage050Bilinear)
{
  testResample("images/index.jpeg", "yaml/index_jpeg_imResample_scale_0_5_method_bilinear_norm_1.yaml");
}

TEST_F(TestUtils, TestResampleColorImage057Bilinear)
{
  testResample("images/index.jpeg", "yaml/index_jpeg_imResample_scale_0_57_method_bilinear_norm_1.yaml");
}

TEST_F(TestUtils, TestResampleColorImage085Bilinear)
{
  testResample("images/index.jpeg", "yaml/index_jpeg_imResample_scale_0_85_method_bilinear_norm_1.yaml");
}

TEST_F(TestUtils, TestResampleColorImage099Bilinear)
{
  testResample("images/index.jpeg", "yaml/index_jpeg_imResample_scale_0_99_method_bilinear_norm_1.yaml");
}


TEST_F(TestUtils, TestResampleConv){
  float I[100]={1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4,4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,6 ,6,6, 6, 6, 6, 6, 6, 6, 6,
  7, 7, 7, 7, 7, 7, 7, 7, 7 ,7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9 ,10, 10, 10, 10, 10, 10, 10, 10, 10, 10 };

  cv::Mat dummy_query = cv::Mat(10,10, CV_32F, I);
  cv::Mat dst;
  cv::Mat output_image = convTri(dummy_query, 3);
  transpose(output_image, output_image);
  cv::Mat img1;
  output_image.convertTo(img1, CV_32F);    
  float *valuesImgConv = img1.ptr<float>();


  FileStorage fs1;
  bool file_exists = fs1.open("yaml/TestConv2.yml", FileStorage::READ);
  ASSERT_TRUE(file_exists);

  FileNode rows = fs1["convTri"]["rows"];
  FileNode cols = fs1["convTri"]["cols"];
  FileNode imgMatlab = fs1["convTri"]["data"];

  for(int i=0;i<(int)rows*(int)cols;i++)
  { 
    ASSERT_TRUE(abs((float)valuesImgConv[i] - (float)imgMatlab[i]) < 0.01);
  }

}

TEST_F(TestUtils, TestResampleConv2)
{
  float I[25] = {1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5};
  cv::Mat dummy_query = cv::Mat(5,5, CV_32F, I);
  cv::Mat dst;

  cv::Mat output_image = convTri(dummy_query, 5);
  transpose(output_image, output_image);
  cv::Mat img1;
  output_image.convertTo(img1, CV_32F);    
  float *valuesImgConv = img1.ptr<float>();

  FileStorage fs1;
  bool file_exists = fs1.open("yaml/TestConv1.yml", FileStorage::READ);
  ASSERT_TRUE(file_exists);

  FileNode rows = fs1["convTri"]["rows"];
  FileNode cols = fs1["convTri"]["cols"];
  FileNode imgMatlab = fs1["convTri"]["data"];


  for(int i=0;i<(int)rows*(int)cols;i++)
  { 
    ASSERT_TRUE(abs((float)valuesImgConv[i] - (float)imgMatlab[i]) < 0.01);
  }


}


TEST_F(TestUtils, TestResampleConvReal)
{
  cv::Mat image = cv::imread("images/imgGrayScale.jpeg", cv::IMREAD_GRAYSCALE);

  cv::Mat output_image = convTri(image, 5);

  transpose(output_image, output_image);
  cv::Mat img1;
  output_image.convertTo(img1, CV_32F);    
  float *valuesImgConv = img1.ptr<float>();


  FileStorage fs1;
  bool file_exists = fs1.open("yaml/TestConvReal.yml", FileStorage::READ);
  ASSERT_TRUE(file_exists);


  FileNode rows = fs1["convTri"]["rows"];
  FileNode cols = fs1["convTri"]["cols"];
  FileNode imgMatlab = fs1["convTri"]["data"];

  for(int i=0;i<(int)rows*(int)cols;i++)
  { 
    ASSERT_TRUE(abs((float)valuesImgConv[i] - (float)imgMatlab[i]) < 0.01);
  }


}
