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


  	if(existClassifier == false){
  		loadValue = false;
  	}else{
      for(int i = 0; i < 14; i++){
      	int rows = (int)classifier[clf_aux[i]]["rows"];
      	int cols = (int)classifier[clf_aux[i]]["cols"];
        cv::FileNode num_classes_data = classifier[clf_aux[i]]["data"];

        std::vector<float> p;
        num_classes_data >> p;

        cv::Mat matrix= cv::Mat::zeros(rows, cols, CV_32F);
        memcpy(matrix.data, p.data(), p.size()*sizeof(float));

        clf.insert({clf_aux[i].c_str(), matrix });
      }
  	}

	std::cout << clf["Cprime"] << std::endl;
	std::cout << clf["Y"] << std::endl;

	return loadValue;
}


