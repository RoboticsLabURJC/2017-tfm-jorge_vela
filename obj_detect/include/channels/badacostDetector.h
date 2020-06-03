/** ------------------------------------------------------------------------
 *
 *  @brief badacostDetector.
 *  @author Jorge Vela
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2020/06/01
 *
 *  ------------------------------------------------------------------------ */


#ifndef BADACOST_DETECTOR
#define BADACOST_DETECTOR

#include <opencv/cv.hpp>
#include <vector>
#include <string>

class BadacostDetector
{
private:


  int m_srhink;
  int m_modelHt;
  int m_modelWd;
  int m_stride;
  int m_cascThr;

  cv::FileStorage m_classifier;

protected:

public:
	BadacostDetector(
            /*int shrink = 2, 
            int modelHt = 2, 
            int modelWd = 2, 
            int stride = 2, 
            int cascThr = 2*/
	){   
            /*m_srhink = shrink;
            m_modelHt = modelHt;
            m_modelWd = modelWd;
            m_stride = stride;
            m_cascThr = cascThr;*/
	};

      bool load(std::string clf);

      std::vector<float> detect(std::vector<cv::Mat> imgs);

};


#endif

