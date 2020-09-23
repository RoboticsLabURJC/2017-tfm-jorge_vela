/** ------------------------------------------------------------------------
 *
 *  @brief Test of Image Resample
 *  @author Jorge Vela Peña
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2020/06/02
 *
 *  ------------------------------------------------------------------------ */


#include <channels/badacostDetector.h> 
#include "gtest/gtest.h"
#include <opencv2/opencv.hpp>

#include <iostream>


class TestBadacostDetector: public testing::Test
{
public:
  BadacostDetector badacost;
  virtual void SetUp()
    {
    }

  virtual void TearDown()
    {
    }
};

TEST_F(TestBadacostDetector, loadClassifier){

  /*float A[] = {1, 2, 1 , 1 ,3 ,1 , 4 ,2, 2};
  float B[] = {1, 2, 2, 1};

  cv::Mat A1 = cv::Mat(3,3, CV_32FC1, A);
  cv::Mat B1 = cv::Mat(2,2, CV_32FC1, B);

  cv::RNG rng(12345);
  cv::Scalar value = cv::Scalar( rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255) );
  //copyMakeBorder( A1, A1, 0,0,1,1, cv::BORDER_CONSTANT, 0 );

  printf("--> %f \n", (float)A1.at<float>(2,0) );
  cv::Mat out_image;
  filter2D(A1, out_image,  CV_32FC1 , B1, cv::Point( 0.0, 0.0 ), 0, cv::BORDER_CONSTANT );

  for(int i = 0; i < 3; i++){
    for(int j = 0; j < 3; j++){
      printf("%f ", (float)out_image.at<float>(i,j));
    }
    printf("\n");
  }

  printf("%d %d \n",out_image.size().width, out_image.size().height );*/

    std::string clfPath = "yaml/clf.yml";
    std::string pyrPath = "yaml/pPyramid.yml";
    std::string filtersPath = "yaml/filterTest.yml";

    bool loadVal = badacost.load(clfPath, pyrPath, filtersPath);
	ASSERT_TRUE(loadVal);

//	cv::Mat image = cv::imread("images/carretera.jpg", cv::IMREAD_COLOR);
    cv::Mat image = cv::imread("images/coches10.jpg", cv::IMREAD_COLOR);

//    cv::imshow("image", image);
//    cv::waitKey();

  std::vector<cv::Rect2i> detections = badacost.detect(image);

  for(uint i = 0; i < detections.size(); i++)
  {
    std::cout << "[ x=" << detections[i].x << ", y=";
    std::cout << detections[i].y << ", w=" << detections[i].width << ", h=" << detections[i].height;
    std::cout << " ] " << std::endl;
    cv::rectangle(image, detections[i], cv::Scalar(200,0,0),2);
  }
  cv::imshow("image", image);
  cv::waitKey();
}














