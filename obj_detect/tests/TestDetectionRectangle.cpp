/** ------------------------------------------------------------------------
 *
 *  @brief Test of Extraction of ACF Channels.
 *  @author Jorge Vela Peña
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *
 *  ------------------------------------------------------------------------ */

#include <detectors/DetectionRectangle.h>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <iostream>



class TestDetectionrectangle: public testing::Test
{
public:
  virtual void SetUp()
    {
    }

  virtual void TearDown()
    {
    }
};

TEST_F(TestDetectionrectangle, TestDetectionrectangleSqarify)
{
	cv::Rect bbox1(10,10,30,30);
	float score = 3;     // Detection score (the higher the more confident).
	int class_index = 2;

	cv::Rect bbox2(10,7,30,25);
	float score2 = 3;     // Detection score (the higher the more confident).
	int class_index2 = 2;

	cv::Rect bbox3(14,10,25,30);
	float score3 = 3;     // Detection score (the higher the more confident).
	int class_index3 = 2;

	cv::Rect bbox4(5,5,8,8);
	float score4 = 3;     // Detection score (the higher the more confident).
	int class_index4 = 2;

	DetectionRectangle d1;
	d1.bbox = bbox1;
	d1.score = score;
	d1.class_index = class_index;
	d1.squarify(0, 2.0);
	ASSERT_TRUE(d1.bbox.x == -5);
	ASSERT_TRUE(d1.bbox.width == 60);
	ASSERT_TRUE(d1.bbox.y == 10);
	ASSERT_TRUE(d1.bbox.height == 30);	

	DetectionRectangle d2;
	d2.bbox = bbox2;
	d2.score = score2;
	d2.class_index = class_index2;
	d2.squarify(1, 2.0);
	ASSERT_TRUE(d2.bbox.x == 10);
	ASSERT_TRUE(d2.bbox.width == 30);
	ASSERT_TRUE(d2.bbox.y == 12);
	ASSERT_TRUE(d2.bbox.height == 15);


	DetectionRectangle d3;
	d3.bbox = bbox3;
	d3.score = score3;
	d3.class_index = class_index3;
	d3.squarify(3, 3.0);
	ASSERT_TRUE(d3.bbox.x == -19);
	ASSERT_TRUE(d3.bbox.width == 90);
	ASSERT_TRUE(d3.bbox.y == 10);
	ASSERT_TRUE(d3.bbox.height == 30);


	DetectionRectangle d4;
	d4.bbox = bbox4;
	d4.score = score4;
	d4.class_index = class_index4;
	d4.squarify(3, 3.0);
	ASSERT_TRUE(d4.bbox.x == -3);
	ASSERT_TRUE(d4.bbox.width == 24);
	ASSERT_TRUE(d4.bbox.y == 5);
	ASSERT_TRUE(d4.bbox.height == 8);

}
