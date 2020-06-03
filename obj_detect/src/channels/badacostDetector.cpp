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
	std::map<std::string, cv::FileNode> clf;
	std::string clf_aux[14] = {"fids", "thrs", "child", "hs", "weights", "depth", "treeDepth", "num_classes", "Cprime", "Y", "w1_weights", "weak_learner_type", "aRatio", "aRatioFixedWidth"};

	cv::FileStorage classifier;
  	bool existClassifier = classifier.open(clfPath, cv::FileStorage::READ);

  	if(existClassifier == false){
  		loadValue = false;
  	}else{
      for(int i = 0; i < 14; i++){
        cv::FileNode num_classes_data = classifier[clf_aux[i]]["rows"];
        if(num_classes_data.size() == 0){loadValue = false; }
        clf.insert({clf_aux[i], num_classes_data });
        printf("%s %d \n", clf_aux[i].c_str() ,(int)num_classes_data);
      }
  	}

	return loadValue;
}