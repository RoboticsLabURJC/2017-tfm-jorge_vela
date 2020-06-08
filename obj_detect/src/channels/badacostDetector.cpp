/** ------------------------------------------------------------------------
 *
 *  @brief badacostDetector.
 *  @author Jorge Vela
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2020/06/01
 *
 *  ------------------------------------------------------------------------ */


#include <channels/badacostDetector.h> 
#include "gtest/gtest.h"
#include <opencv/cv.hpp>

#include <iostream>

bool BadacostDetector::load(std::string clfPath){
	bool loadValue = true;

	std::string clf_aux[14] = {"fids", "thrs", "child", "hs", "weights", "depth", "treeDepth", "num_classes", "Cprime", "Y", "w1_weights", "weak_learner_type", "aRatio", "aRatioFixedWidth"};

	cv::FileStorage classifier;
  	bool existClassifier = classifier.open(clfPath, cv::FileStorage::READ);


	std::map<std::string, cv::Mat> clf;
	std::vector<cv::Mat> clf2;
	cv::Mat prueba;
	std::vector<float> p;


    printf("%d\n", clf2.size());

  	if(existClassifier == false){
  		loadValue = false;
  	}else{
      for(int i = 8; i < 9; i++){
      	int rows = (int)classifier[clf_aux[i]]["rows"];
      	int cols = (int)classifier[clf_aux[i]]["cols"];
        cv::FileNode num_classes_data = classifier[clf_aux[i]]["data"];


        num_classes_data >> p;

        float *p2 = &p[0];

        printf("%s --> %.4f %d %d\n",clf_aux[i].c_str(), p2[0] , rows, cols);

        cv::Mat mat = cv::Mat(rows,cols, CV_32F, p2);
        //(float*)num_classes_data;
        //if(num_classes_data.size() == 0){loadValue = false; }
        clf.insert({clf_aux[i].c_str(), mat });
      } 
  	}

	std::cout << clf["Cprime"] << std::endl;
	//std::cout << "M = " << std::endl << " "  << x << std::endl << std::endl;


	return loadValue;
}


