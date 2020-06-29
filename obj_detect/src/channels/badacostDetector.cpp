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

        cv::Mat matrix= cv::Mat::zeros(cols, rows, CV_32F);
        memcpy(matrix.data, p.data(), p.size()*sizeof(float));

        clf.insert({clf_aux[i].c_str(), matrix });
      }
  	}

  	m_classifier = clf;
	return loadValue;
}


std::vector<float> BadacostDetector::detect(cv::Mat imgs){
	Utils utils;
	ChannelsPyramid chnsPyramid;

	//CARGO LOS PARAMETROS, LLAMO A CHNSPYRAMID, SE PASA TODO POR EL FILTRO Y SE HACE RESIZE. 
	//EQUIVALENTE HASTA LINEA 80 acfDetectBadacost. Mismos resultados aparentemente.
	std::string nameOpts = "yaml/pPyramid.yml";
  	bool loadOk = chnsPyramid.load(nameOpts.c_str());
	std::vector<cv::Mat> pyramid = chnsPyramid.getPyramid(imgs);

	std::vector<cv::Mat> filteredImages = chnsPyramid.badacostFilters(pyramid, "yaml/filterTest.yml");

 	std::vector<cv::Mat> filteredImagesResized;
	for(int i = 0; i < filteredImages.size(); i++){
		cv::Mat imgResized = utils.ImgResample(filteredImages[i], filteredImages[i].size().width/2, filteredImages[i].size().height/2 );
		filteredImagesResized.push_back(imgResized);
	}
      


	//COMIENZA EL SEGUNDO BUCLE
    int modelDsPad[2] = {54, 96};
	int modelDs[2] = {48, 84};

    int shrink = 4;
    int modelHt = 54;
    int modelWd = 96;
    int stride = 4;
    float cascThr = 1; 


    int num_classes = m_classifier["num_classes"].at<float>(0,0);
    printf("%d\n", num_classes);

    int height = filteredImagesResized[0].size().height;
    int width = filteredImagesResized[0].size().width;
    int nChan = filteredImagesResized.size();


	int nTreeNodes = m_classifier["fids"].size().width;
	int nTrees = m_classifier["fids"].size().height;
	int height1 = ceil(float(height*shrink-modelHt+1)/stride);
	int width1 = ceil(float((width*shrink)-modelWd+1)/stride);
	
	std::vector<double> margin_vector(num_classes);
	std::vector<double> costs_vector(num_classes);




    int nFtrs = modelHt/shrink*modelWd/shrink*nChan;
    int *cids = new int[nFtrs]; //LO TIENE COMO UINT32 ... 

    int m=0;
    for( int z=0; z<nChan; z++ )
      for( int c=0; c<modelWd/shrink; c++ )
        for( int r=0; r<modelHt/shrink; r++ )
          cids[m++] = z*width*height + c*height + r;


	int num_windows = width1*height1;
	if (num_windows < 0)  // Detection window is too big for the image 
	  num_windows = 0;   // Detect on 0 windows in this case (do nothing).

    std::vector<int> rs(num_windows), cs(num_windows); 
    std::vector<float> hs1(num_windows), scores(num_windows);

  	for( int c=0; c<width1; c++ ){
  	  for( int r=0; r<height1; r++ ) { 
  	  	//if(c == 0 && r == 0){
  	  	//	printf("%f \n", filteredImagesResized[0].at<float>(0,0) );
  	  	//}
  	    std::vector<double> margin_vector(num_classes);
        double trace;
        int h;

        //float *chns1=chns+(r*stride/shrink) + (c*stride/shrink)*height;
        int posHeight = (r*stride/shrink);
	    int posWidth = (c*stride/shrink);
        // Initialise the margin_vector memory to 0.0
        for(int i=0; i<num_classes; i++)
        {
          margin_vector[i] = 0.0;
        }

       for(int t = 0; t < nTrees; t++ ) 
       {
          int offset=t*nTreeNodes, k=offset, k0=k;

          int v1 = int(k/869);
          int v2 = int(k % 869);
          //printf("%d %d\n",v1,v2 );
          int childVal = (int)m_classifier["child"].at<float>(v1, v2);
          while(childVal){
          //if(t==0 || t == 1){
            int ftrCids = cids[(int)m_classifier["fids"].at<float>(v1, v2)];

            int ftrHeight = ftrCids / width1;
            int ftrWidth = ftrCids % width1;


            int ftr = filteredImagesResized[0].at<float>(ftrHeight + v1, ftrWidth + v2);
            printf("-->%d %d\n", v1, v2 );
            int thrs = (int)m_classifier["thrs"].at<float>(v1,v2);


            k = (ftr<thrs) ? 1 : 0;

            int v1_k0 = int(k0/869);
            int v2_k0 = int(k0/869);


            k0 = k = (int)m_classifier["child"].at<float>(v1_k0, v2_k0)-k+offset;



            v1 = int(k/869);
            v2 = int(k % 869);
            childVal = (int)m_classifier["child"].at<float>(v1, v2);
          	//printf("%d %d %d \n",  (int)m_classifier["fids"].at<float>(v1, v2), ftrHeight, ftrWidth);
          }

          //h = static_cast<int>((int)m_classifier["hs"].at<float>(v1, v2));

          /*for(int i=0; i<num_classes; i++)
          {
          	//codified[i]
          	int valCod = h + i;
            int v1y = static_cast<size_t>(num_classes*(h-1)/21);
            int v2y = static_cast<size_t>(num_classes*(h-1)%21);
            double codified = m_classifier["Y"].at<float>(v1y, v2y); 

            //wl_weights[t];
            margin_vector[i] += codified * m_classifier["w1_weights"].at<float>(0, t) ;
          }    

          double min_pos_cost;
          double neg_cost;
          // Gets positive class min cost and label in h!        
          getMinPositiveCost(num_classes, m_classifier["Cprime"], margin_vector, min_pos_cost, h);
          getNegativeCost(num_classes, Cprime, margin_vector, neg_cost);
          trace = -(min_pos_cost - neg_cost);
	        
          if (trace <=cascThr) break;*/ 
  	   }


  	  /*
      if (trace < 0) h=1;
   
      if (h > 1) 
      {
        int index = c + (r*width1);
        cs[index] = c; 
        rs[index] = r; 
        hs1[index] = h; 
        scores[index] = trace; 
      }
      delete [] cids; 

	  int size_output=0;
	  for( int i=0; i<hs1.size(); i++ ) {
	    if (hs1[i] > 1) 
	    {
	      size_output++;
	    }
	  }*/


      }
	}
	std::vector<float> del;
	return del;
}




























