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
#include <opencv/cv.hpp>

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

  /*float A[] = {1, 2, 1,3 , 1 ,3 ,1,3 , 4 ,2, 2,3,4,3,6,2};
  float B[] = {1, 2, 2, 1};

  cv::Mat A1 = cv::Mat(4,4, CV_32FC1, A);
  cv::Mat B1 = cv::Mat(2,2, CV_32FC1, B);

  cv::RNG rng(12345);
  cv::Scalar value = cv::Scalar( rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255) );
  //copyMakeBorder( A1, A1, 0,0,1,1, cv::BORDER_CONSTANT, 0 );

  cv::Mat out_image;
  filter2D(A1, out_image,  CV_32FC1 , B1, cv::Point( 0.0, 0.0 ), 0, cv::BORDER_CONSTANT );

  for(int i = 0; i < out_image.size().height; i++){
    for(int j = 0; j < out_image.size().width; j++){
      printf("%f ", (float)out_image.at<float>(i,j));
    }
    printf("\n");
  }
  
  printf("%d %d \n",out_image.size().width, out_image.size().height );
  */


	std::string clfPath = "yaml/clf.yml";
	bool loadVal = badacost.load(clfPath.c_str());
	ASSERT_TRUE(loadVal);

	cv::Mat image = cv::imread("images/carretera.jpg", cv::IMREAD_COLOR);
  

  //std::vector<cv::Rect2i> detections = 
	badacost.detect(image);  

  //for(int i = 0; i < detections.size(); i++)
  //  rectangle(image,detections[i],cv::Scalar(200,0,0),2);
  //cv::imshow("image", image);
  //cv::waitKey();
  
  
}














