/** ------------------------------------------------------------------------
 *
 *  @brief badacostDetector.
 *  @author Jorge Vela
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2020/06/01
 *
 *  ------------------------------------------------------------------------ */


#include <channels/badacostDetector.h> 
#include <channels/ChannelsPyramid.h>
#include "gtest/gtest.h"
#include <opencv/cv.hpp>
#include <channels/Utils.h>

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

  	m_classifier = clf;
	return loadValue;
}


std::vector<float> BadacostDetector::detect(cv::Mat imgs){
	//std::cout << m_classifier["Cprime"] << std::endl;
	//DISTINTOS DATOS DEL CLASIFICADOR 
	float thrs = m_classifier["thrs"].at<float>(0,0);
	float hs = m_classifier["hs"].at<float>(0,0);
	float fids = m_classifier["fids"].at<float>(0,0);
	float child = m_classifier["child"].at<float>(0,0);
	float treeDepth = m_classifier["treeDepth"].at<float>(0,0);
	float Cprime = m_classifier["Cprime"].at<float>(0,0);
	float Y = m_classifier["Y"].at<float>(0,0);
	float w1_weights = m_classifier["w1_weights"].at<float>(0,0);
	float num_classes = m_classifier["num_classes"].at<float>(0,0);


	//printf("%.4f %.4f %.4f %.4f %.4f \n", thrs, hs, fids, child, treeDepth);

	cv::Size s = imgs.size();
	int rows = s.height;
	int cols = s.width;
	printf("%d %d \n",rows, cols );


    //LLAMAR A CHNSPYRAMID CON LA IMAGEN PARA OBTENER TODAS LAS ESCALAS
    ChannelsPyramid chnsPyramids;
    chnsPyramids.load("yaml/pPyramid.yml");
	std::vector<cv::Mat> pyramid = chnsPyramids.getPyramid(imgs);

	//printf("%d\n", pyramid.size());

	//HACER UN BUCLE UTILIZANDO getSingleScale PARA OBTENER LOS VALORES EN CADA ESCALA
	for(int i = 0; i < pyramid.size(); i++){
		//LLAMAMOS A DETECTSINGLESCALE
	}


	/*for(int c = 0; c < rows; c++){
		for(int r = 0; r < rows; r++){
			std::vector<double> margin_vector(num_classes);


		}
	}*/




	std::vector<float> del;
	return del;
}












