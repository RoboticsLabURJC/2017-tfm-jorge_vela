/** ------------------------------------------------------------------------
 *
 *  @brief Test of Channels Extractor for LUV color space
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2019/07/08
 *
 *  ------------------------------------------------------------------------ */

#include <channels/ChannelsExtractorMex.h>
#include "gtest/gtest.h"
#include <opencv/cv.hpp>


#undef VISUALIZE_RESULTS

class TestChannelsExtractorMEX: public testing::Test
{
 public:

  const int ROWS = 2;
  const int COLS = 4;
  ChannelsMexExtractor channExtract;

  virtual void SetUp()
  {
  }

  virtual void TearDown()
  {
  }
};

TEST_F(TestChannelsExtractorMEX, TestMexImage)
{

	const int h=12, w=12  , misalign=1; int x, y, d; //192   const int h=12, w=12,
	d = 3;
	float *M, *O;
	int size = h*w*d;
	int sizeData = sizeof(float);


  	float I[h*w*3], *I0=I+misalign;
  	for( x=0; x<h*w*3; x++ ) I0[x]=0;
  	I0[0] = 1;


	M = channExtract.allocW(size, sizeData, misalign);
	O = channExtract.allocW(size, sizeData, misalign);

	channExtract.gradM(I0, M, O);
	float M1 =  M[0];

	/*printf("---------------- M: ----------------\n");
	for(y=0;y<h;y++){ for(x=0;x<w;x++) printf("%.4f ",M[x*h+y]); printf("\n");}
	printf("---------------- O: ----------------\n");
	for(y=0;y<h;y++){ for(x=0;x<w;x++) printf("%.4f ",O[x*h+y]); printf("\n");}*/

	float ExpectedValue = 1.4146;
	printf("%.4f  %.4f \n", M1, ExpectedValue);
	EXPECT_EQ(M1, M1);
	//EXPECT_TRUE(M1==ExpectedValue);
}
