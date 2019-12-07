/** ------------------------------------------------------------------------
 *
 *  @brief Test of Channels Extractor for LUV color space
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2019/07/08
 *
 *  ------------------------------------------------------------------------ */

#include <channels/ChannelsExtractorGradMag.h>
#include "gtest/gtest.h"
#include <opencv/cv.hpp>


#undef VISUALIZE_RESULTS

class TestChannelsExtractorGradMag: public testing::Test
{
 public:

  const int ROWS = 2;
  const int COLS = 4;
  GradMagExtractor gradMagExtract;

  virtual void SetUp()
  {
  }

  virtual void TearDown()
  {
  }
};

TEST_F(TestChannelsExtractorGradMag, TestGradMagUpRight)
{

	const int h=12, w=12  , misalign=1; int x, y, d; //192   const int h=12, w=12,
	d = 3;
	float *M, *O, *H;
	int size = h*w*d;
	int sizeData = sizeof(float);


  	float I[h*w*3], *I0=I+misalign;
  	for( x=0; x<h*w*3; x++ ) I0[x]=0;
  	I0[0] = 1;
  
	M = gradMagExtract.allocW(size, sizeData, misalign);
	O = gradMagExtract.allocW(size, sizeData, misalign);

	gradMagExtract.gradM(I0, M, O);

	ASSERT_TRUE(abs(M[0] - 1.4146) < 1.e-4f);
	ASSERT_TRUE(abs(M[1] - 0.5001) < 1.e-4f);
	ASSERT_TRUE(abs(M[12] - 0.5001) < 1.e-4f);

	ASSERT_TRUE(abs(O[0] - 0.7857) < 1.e-4f);
	ASSERT_TRUE(abs(O[1] - 1.5708) < 1.e-4f);
	ASSERT_TRUE(abs(O[12] - 3.1171) < 1.e-4f);
}

TEST_F(TestChannelsExtractorGradMag, TestGradMagDownLeft)
{

	const int h=12, w=12  , misalign=1; int x, y, d; //192   const int h=12, w=12,
	d = 3;
	float *M, *O, *H;
	int size = h*w*d;
	int sizeData = sizeof(float);


  	float I[h*w*3], *I0=I+misalign;
  	for( x=0; x<h*w*3; x++ ) I0[x]=0;
  	I0[143] = 1;
  
	M = gradMagExtract.allocW(size, sizeData, misalign);
	O = gradMagExtract.allocW(size, sizeData, misalign);

	gradMagExtract.gradM(I0, M, O);


	ASSERT_TRUE(abs(M[143] - 1.4146) < 1.e-4f);
	ASSERT_TRUE(abs(M[142] - 0.5001) < 1.e-4f);
	ASSERT_TRUE(abs(M[131] - 0.5001) < 1.e-4f);

	ASSERT_TRUE(abs(O[143] - 0.7857) < 1.e-4f);
	ASSERT_TRUE(abs(O[142] - 1.5708) < 1.e-4f);
	ASSERT_TRUE(abs(O[131] - 0.0245) < 1.e-4f);
}


TEST_F(TestChannelsExtractorGradMag, TestGradMagCenter)
{

	const int h=12, w=12  , misalign=1; int x, y, d; //192   const int h=12, w=12,
	d = 3;
	float *M, *O, *H;
	int size = h*w*d;
	int sizeData = sizeof(float);


  	float I[h*w*3], *I0=I+misalign;
  	for( x=0; x<h*w*3; x++ ) I0[x]=0;
  	I0[76] = 1;
  
	M = gradMagExtract.allocW(size, sizeData, misalign);
	O = gradMagExtract.allocW(size, sizeData, misalign);

	gradMagExtract.gradM(I0, M, O);

	ASSERT_TRUE(abs(M[77] - 0.5001) < 1.e-4f);
	ASSERT_TRUE(abs(M[75] - 0.5001) < 1.e-4f);
	ASSERT_TRUE(abs(M[64] - 0.5001) < 1.e-4f);
	ASSERT_TRUE(abs(M[88] - 0.5001) < 1.e-4f);


	ASSERT_TRUE(abs(O[77] - 1.5708) < 1.e-4f);
	ASSERT_TRUE(abs(O[75] - 1.5708) < 1.e-4f);
	ASSERT_TRUE(abs(O[64] - 0.0245) < 1.e-4f);
	ASSERT_TRUE(abs(O[88] - 3.1171) < 1.e-4f);
}

TEST_F(TestChannelsExtractorGradMag, TestGradMagLinear)
{
	const int h=12, w=12  , misalign=1; int x, y, d; //192   const int h=12, w=12,
	d = 3;
	float *M, *O, *H;
	int size = h*w*d;
	int sizeData = sizeof(float);


  	float I[h*w*3], *I0=I+misalign;
  	for( x=0; x<h*w*3; x++ ) I0[x]=0;
  	for(x = 60; x < 72; x++) I0[x] = 1;
  
	M = gradMagExtract.allocW(size, sizeData, misalign);
	O = gradMagExtract.allocW(size, sizeData, misalign);

	gradMagExtract.gradM(I0, M, O);


	for(y=48; y < 59;  y++){
		ASSERT_TRUE(abs(M[y] - 0.5001) < 1.e-4f);
	}

	for(y=72; y < 83;  y++){
		ASSERT_TRUE(abs(M[y] - 0.5001) < 1.e-4f);
	}

	for(y=48; y < 59;  y++){
		ASSERT_TRUE(abs(O[y] - 0.0245) < 1.e-4f);
	}

	for(y=72; y < 83;  y++){
		ASSERT_TRUE(abs(O[y] - 3.1171) < 1.e-4f);
	}
}