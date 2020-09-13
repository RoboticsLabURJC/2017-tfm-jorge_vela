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

typedef unsigned int uint32;

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
        //printf("%d\n",i );
        int rows = (int)classifier[clf_aux[i]]["rows"];
        int cols = (int)classifier[clf_aux[i]]["cols"];
        cv::FileNode num_classes_data = classifier[clf_aux[i]]["data"];

        std::vector<float> p;
        num_classes_data >> p;

        cv::Mat matrix= cv::Mat::zeros(cols, rows, CV_32F);
        memcpy(matrix.data, p.data(), p.size()*sizeof(float));
        clf.insert({clf_aux[i].c_str(), matrix });
        //printf("%s\n",clf_aux[i].c_str() );
      }
    }

  m_classifier = clf;
  return loadValue;
}


inline void getNegativeCost(int num_classes, float *Cprime, 
        std::vector<float>& margin_vector, float& neg_cost)
{
  // The index of the negative class is assumed 1. Therefore, its
  // column in Cprime is the first one (no need to move to its column).
  neg_cost = 0.0;
  float* cprime_column = Cprime;
  for(int i=0; i < num_classes; i++)
  {
    neg_cost += cprime_column[i] * margin_vector[i];
  }
}


inline void getMinPositiveCost(int num_classes, 
                               float *Cprime, 
                               std::vector<float>& margin_vector, 
                               float& min_value, 
                               int& h)
{
  min_value = std::numeric_limits<double>::max();
  for(int j=1; j < num_classes; j++)
  {
    float cost = 0.0;
    float* cprime_column = Cprime + static_cast<size_t>(j*num_classes);
    for(int i=0; i < num_classes; i++)
    {
      cost += cprime_column[i] * margin_vector[i];
    }

    if (cost < min_value) 
    {
      min_value = cost;
      h = j+1;
    }
  }        
}



std::vector<cv::Rect2i> BadacostDetector::detect(cv::Mat imgs){
  Utils utils;


  ChannelsPyramid chnsPyramid;
  //CARGO LOS PARAMETROS, LLAMO A CHNSPYRAMID, SE PASA TODO POR EL FILTRO Y SE HACE RESIZE. 
  //EQUIVALENTE HASTA LINEA 80 acfDetectBadacost. Mismos resultados aparentemente.
  std::string nameOpts = "yaml/pPyramid.yml";
  bool loadOk = chnsPyramid.load(nameOpts.c_str());
  std::vector<cv::Mat> pyramid = chnsPyramid.getPyramid(imgs);

  ///int nChannels = pyramid[0].channels();
  //cv::Mat bgr_dst[nChannels];
  //split(pyramid[0],bgr_dst);

  for(int i = 0; i < 1 /*pyramid.size()*/; i++){
    cv::Mat dst;
    cv::RNG rng(12345);

    cv::Scalar value = cv::Scalar( rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255) );
    copyMakeBorder( pyramid[i], dst, 2, 2, 3, 3, cv::BORDER_REPLICATE, value );
    std::vector<cv::Mat> filteredImagesResized /*filteredImages*/ = chnsPyramid.badacostFilters(dst, "yaml/filterTest.yml");

    printf("%d %d \n", filteredImagesResized[0].size().width, filteredImagesResized[0].size().height);

    //ESTO SE HA AÑADIDO EN BADACOST FILTERS QUE ES EN REALIDAD DONDE SE HACE EN MATLAB
    //std::vector<cv::Mat> filteredImagesResized;
    //for(int i = 0; i < filteredImages.size(); i++){
    //cv::Mat imgResized = utils.ImgResample(filteredImages[i], filteredImages[i].size().width/2, filteredImages[i].size().height/2 );
    //filteredImagesResized.push_back(imgResized);
    //}

    
    //COMIENZA EL SEGUNDO BUCLE
    int modelDsPad[2] = {54, 96}; //MISMO ORDEN QUE MATLAB = 54, 96
    int modelDs[2] = {48, 84}; //MISMO ORDEN QUE MATLAB = 48, 84
    int shrink = 4;
    float cascThr = -2.23; 
    int stride = 4;
    
    
    int num_classes = m_classifier["num_classes"].at<float>(0,0);
    int nTreeNodes = m_classifier["fids"].size().width;
    int nTrees = m_classifier["fids"].size().height;

    float *child = m_classifier["child"].ptr<float>();
    float *thrs = m_classifier["thrs"].ptr<float>();
    float *fids = m_classifier["fids"].ptr<float>();
    float *hs = m_classifier["hs"].ptr<float>();
    float *Y = m_classifier["Y"].ptr<float>();
    float *w1_weights = m_classifier["w1_weights"].ptr<float>();
    float *Cprime = m_classifier["Cprime"].ptr<float>();


    int height = filteredImagesResized[0].size().height;
    int width = filteredImagesResized[0].size().width;
    int nChan = filteredImagesResized.size();
   

    int modelHt = modelDsPad[0];
    int modelWd = modelDsPad[1];
    int height1 = ceil(float(height*shrink-modelHt+1)/stride);
    int width1 = ceil(float((width*shrink)-modelWd+1)/stride);

    int nFtrs = modelHt/shrink*modelWd/shrink*nChan;


    uint32 *cids = new uint32[nFtrs]; //LO TIENE COMO UINT32 ... 


    uint32 *zsA = new uint32[nFtrs];
    uint32 *csA = new uint32[nFtrs];
    uint32 *rsA = new uint32[nFtrs];
    int m=0;
    for( int z=0; z<nChan; z++ )
      for( int c=0; c<modelWd/shrink; c++ )
        for( int r=0; r<modelHt/shrink; r++ ){
          //if(z*width*height + c*height + r == 389760)
          //  printf("%d %d %d \n", z,c,r);
          zsA[m] = z; 
          csA[m] = c;
          rsA[m] = r;
          cids[m++] = z*width*height + c*height + r;

        }


    int num_windows = width1*height1;


    std::vector<int> rs(num_windows), cs(num_windows); 
    std::vector<float> hs1(num_windows), scores(num_windows);


    //printf("%f \n", (float)filteredImagesResized[7].at<float>(12,14));
    cv::Mat finalFiltered;
    merge(filteredImagesResized, finalFiltered);
    float *chns = finalFiltered.ptr<float>();

    for( int c=0; c < 1 /*width1*/; c++ ){
      for( int r=0; r < 1 /*height1*/; r++ ) { 
        std::vector<float> margin_vector(num_classes);
        double trace;
        int h;
        //float *chns1=chns+(r*stride/shrink) + (c*stride/shrink)*height;
        
        // Initialise the margin_vector memory to 0.0
        for(int i=0; i<num_classes; i++)
        {
          margin_vector[i] = 0.0;
        }


        for(int t = 0; t < nTrees; t++ ) 
        {
          int offset=t*nTreeNodes, k=offset, k0=k;
          int val = (int)fids[offset];

          while(child[k]){
            //printf("cid %d \n",cids[(int)fids[k]] );

            int chanSearch = zsA[(int)fids[k]];
            int hSearch = csA[(int)fids[k]];
            int wSearch = rsA[(int)fids[k]];
            //printf("%d %d %d \n",chanSearch, hSearch, wSearch );

            float ftr = (float)filteredImagesResized[chanSearch].at<float>(wSearch,hSearch);//chns[cids[(int)fids[k]]];
            //printf("ftr: %f \n",ftr );
            k = (ftr<thrs[k]) ? 1 : 0;
            //printf("k: %d \n", k  );
            k0 = k = child[k0]-k+offset;
            //printf("k0:  %d \n", k0 ); 
          }
          //printf("--------------------------\n");
          h = static_cast<int>(hs[k]);

          float* codified = Y + static_cast<size_t>(num_classes*(h-1));

          printf("%d\n",num_classes );
          for(int i=0; i<num_classes; i++)
          {
              margin_vector[i] += codified[i] * w1_weights[t];
              //printf("%f %f \n",codified[i], w1_weights[t] );
          }

          float min_pos_cost;
          float neg_cost;

          getMinPositiveCost(num_classes, Cprime, margin_vector, min_pos_cost, h);
          getNegativeCost(num_classes, Cprime, margin_vector, neg_cost);

          trace = -(min_pos_cost - neg_cost);


          //printf("%f %f %f \n", min_pos_cost, neg_cost, trace );

          if (trace <=cascThr) break; 


        }
        if (trace < 0) h=1;
        
        if(h != 1)
          printf("%d\n",h );

        if (h > 1) 
        {
          int index = c + (r*width1);
          cs[index] = c; 
          rs[index] = r; 
          hs1[index] = h; 
          scores[index] = trace; 
        }
      }
    }
    int size_output=0;
    for( int i=0; i<hs1.size(); i++ ) {
      if (hs1[i] > 1) 
      {
        size_output++;
      }
    }

    delete [] cids; 
    std::vector<cv::Rect2i> detections;
    for( int i=0; i<hs1.size(); i++ ) 
    {
      if (hs1[i] > 1) // hs1[i]>1 are object windows, hs1[i]==1 are background windows.
      {
        //printf("%f\n", hs1[i] );
          printf("%d %d %d %d \n", cs[i] * stride, cs[i],rs[i] * stride, rs[i]);
          cv::Rect2i rect;
          rect.x = cs[i] * stride;
          rect.y = rs[i] * stride;
          rect.width = modelWd;
          rect.height = modelHt;
          
          detections.push_back(rect);
      }
    }
    printf("%d\n",detections.size());


    /*
    std::vector<double> margin_vector(num_classes);
    std::vector<double> costs_vector(num_classes);


    //printf("%d %d %d w: %d h: %d \n", nChan, modelWd/shrink, modelHt/shrink , width, height);

    
    if (num_windows < 0)  // Detection window is too big for the image 
      num_windows = 0;   // Detect on 0 windows in this case (do nothing).

    //std::vector<float> vec;
    //m_classifier["child"].col(0).copyTo(vec);
    //float *data = m_classifier["child"].ptr<float>();
    //int prueba = 2610;
    //int v1prueba = prueba/869;
    //int v2prueba = prueba%869; 
    //printf("%f ------ %f \n", data[prueba], (float)m_classifier["child"].at<float>(v1prueba, v2prueba));




    cv::Mat finalFiltered;
    merge(filteredImagesResized, finalFiltered);
    float *chns = finalFiltered.ptr<float>();

    for( int c=0; c<width1; c++ ){
      for( int r=0; r<height1; r++ ) { 
        float *chns1=chns+(r*stride/shrink) + (c*stride/shrink)*height;
        //if(c == 0 && r == 0){
        //  printf("%f \n", filteredImagesResized[0].at<float>(0,0) );
        //}
        std::vector<float> margin_vector(num_classes);
        float trace;
        int h;

        // Initialise the margin_vector memory to 0.0
        for(int i=0; i<num_classes; i++)
        {
          margin_vector[i] = 0.0;
        }


       for(int t = 0; t < nTrees; t++ ) 
       {
          int offset=t*nTreeNodes, k=offset, k0=k;

          int val = (int)fids[offset];
          while(child[k]){
            float ftr = chns1[cids[(int)fids[k]]];
            k = (ftr<thrs[k]) ? 1 : 0;
            k0 = k = child[k0]-k+offset;
          }
        
          h = static_cast<int>(hs[k]);
          float* codified = Y + static_cast<size_t>(num_classes*(h-1));
          for(int i=0; i<num_classes; i++)
          {
              margin_vector[i] += codified[i] * w1_weights[t];
          }

          float min_pos_cost;
          float neg_cost;

          getMinPositiveCost(num_classes, Cprime, margin_vector, min_pos_cost, h);
          getNegativeCost(num_classes, Cprime, margin_vector, neg_cost);
          trace = -(min_pos_cost - neg_cost);
          if (trace <=cascThr) break; 
        }


      if (trace < 0) h=1;
      
      if (h > 1) 
      {
        int index = c + (r*width1);
        cs[index] = c; 
        rs[index] = r; 
        hs1[index] = h; 
        scores[index] = trace; 
      }
      }
    }
    //delete [] cids; 

    int size_output=0;
    for( int i=0; i<hs1.size(); i++ ) {
      if (hs1[i] > 1) 
      {
        size_output++;
      }
    }

    delete [] cids; 
    std::vector<cv::Rect2i> detections;
    for( int i=0; i<hs1.size(); i++ ) 
    {
      if (hs1[i] > 1) // hs1[i]>1 are object windows, hs1[i]==1 are background windows.
      {
        //printf("%f\n", hs1[i] );
          printf("%d %d %d %d \n", cs[i] * stride, cs[i],rs[i] * stride, rs[i]);
          cv::Rect2i rect;
          rect.x = cs[i] * stride;
          rect.y = rs[i] * stride;
          rect.width = modelWd;
          rect.height = modelHt;
          
          detections.push_back(rect);
      }
    }
    printf("--> %d %d \n", filteredImagesResized[0].size().height, filteredImagesResized[0].size().width);

    printf("%d \n", detections.size() );
    //std::max_element(filteredImagesResized[1].begin(),filteredImagesResized[1].end());
    //std::max_element(filteredImagesResized[2].begin(),filteredImagesResized[2].end());
    //std::max_element(filteredImagesResized[3].begin(),filteredImagesResized[3].end());
    //for(int i = 0; i < detections.size(); i++)
    //  rectangle(filteredImagesResized[0],detections[i],cv::Scalar(200,0,200),50);
    
    //cv::imshow("image2", filteredImagesResized[1]);
    //cv::imshow("image3", filteredImagesResized[2]);
    //cv::imshow("image4", filteredImagesResized[3]);
    //cv::imshow("image5", filteredImagesResized[4]);
    //cv::imshow("image6", filteredImagesResized[5]);
    //cv::imshow("image7", filteredImagesResized[6]);
    //cv::imshow("image8", filteredImagesResized[7]);
    //cv::imshow("image9", filteredImagesResized[8]);
    //cv::imshow("image10", filteredImagesResized[9]);

    //cv::waitKey();*/
  }
  std::vector<cv::Rect2i> detectionsfalse;
  return detectionsfalse;
}




























