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

  cv::Mat dst = ImgResample(image, w, h);
  transpose(dst, dst); // JM: Si esto es necesario es porque está guardado mal en el YAML.

  cv::Mat bgr_dst[3];
  split(dst, bgr_dst);

  cv::Mat bgr_resample[3];
  split(imageMatlab, bgr_resample);
  
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
      if(abs(data[i + j] -  (float)imReample3[i + j])> 1)
      {
        difPixels = difPixels + 1;
      }
    }
  }

  ASSERT_TRUE(difPixels < 10);

  cv::Mat bgr_dst1_chng;
  bgr_dst[1].convertTo(bgr_dst1_chng, CV_32F);
  float *data2 = bgr_dst1_chng.ptr<float>();

  int difPixels2 = 0;
  for(int j = 0; j < (int)cols; j++)
  {
    for(int i = 0; i < (int)rows; i++)  
    {
      if(abs(data2[i + j] -  (float)imReample2[i + j])> 1)
      {
        difPixels2 = difPixels2 + 1;
      }
    }
  }


  ASSERT_TRUE(difPixels2 < 1);

  cv::Mat bgr_dst2_chng;
  bgr_dst[2].convertTo(bgr_dst2_chng, CV_32F);
  float *data3 = bgr_dst2_chng.ptr<float>();

  int difPixels3 = 0;
  for(int j = 0; j < (int)cols; j++)
  {
    for(int i = 0; i < (int)rows; i++)
    {
      if(abs(data3[i + j] -  (float)imReample1[i + j])> 1)
      {
        difPixels3 = difPixels3 + 1;
      }
    }
  }

  ASSERT_TRUE(difPixels3 < 1);


  int difPixels4 = 0;
  for(int j = 0; j < (int)cols; j++)
  {
    for(int i = 0; i < (int)rows; i++)
    {
      if(abs(data3[i + j] -  (float)imReample1[i + j])> 1 or abs(data2[i + j] -  (float)imReample2[i + j])> 1 or abs(data[i + j] -  (float)imReample3[i + j])> 1) 
      {
        difPixels4 = difPixels4 + 1;
      }
    }
  }

  ASSERT_TRUE(difPixels4 < 1);
}

TEST_F(TestUtils, TestResampleConv)
{
  cv::Mat image = cv::imread("images/index3.jpeg", cv::IMREAD_GRAYSCALE);
  cv::Mat imgConv = convTri(image, 5);

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
    ASSERT_TRUE(abs((float)valuesImgConv[i] - (float)imgMatlab[i]) < 0.6);
  }
}

