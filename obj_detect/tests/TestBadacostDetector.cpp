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
	std::string clfPath = "yaml/clf.yml";
	bool loadVal = badacost.load(clfPath.c_str());
	ASSERT_TRUE(loadVal);

}














