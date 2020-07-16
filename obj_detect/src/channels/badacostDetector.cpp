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



std::vector<cv::Rect2i> BadacostDetector::detect(cv::Mat imgs){
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

  printf("%d %d \n", height, width);
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


    printf("%d %d %d %d %d \n", nChan, modelWd/shrink, modelHt/shrink , width, height);

  int num_windows = width1*height1;
  if (num_windows < 0)  // Detection window is too big for the image 
    num_windows = 0;   // Detect on 0 windows in this case (do nothing).

  std::vector<int> rs(num_windows), cs(num_windows); 
  std::vector<int> hs1(num_windows), scores(num_windows);



  /*
  std::vector<float> vec;
  m_classifier["child"].col(0).copyTo(vec);
  float *data = m_classifier["child"].ptr<float>();
  int prueba = 2610;
  int v1prueba = prueba/869;
  int v2prueba = prueba%869; 
  printf("%f ------ %f \n", data[prueba], (float)m_classifier["child"].at<float>(v1prueba, v2prueba));

  float *child = m_classifier["child"].ptr<float>();
  float *thrs = m_classifier["thrs"].ptr<float>();
  float *fids = m_classifier["fids"].ptr<float>();
  float *hs = m_classifier["hs"].ptr<float>();
  float *Y = m_classifier["Y"].ptr<float>();
  float *w1_weights = m_classifier["w1_weights"].ptr<float>();
  float *Cprime = m_classifier["Cprime"].ptr<float>();
  */ 
  cv::Mat finalFiltered;
  merge(filteredImagesResized, finalFiltered); 
  float *chns = finalFiltered.ptr<float>();


  for( int c=0; c<width1; c++ ){
    for( int r=0; r<height1; r++ ) { 
      //float *chns1=chns+(r*stride/shrink) + (c*stride/shrink)*height;
      int v = (r*stride/shrink) + (c*stride/shrink)*height;
      //if(c == 0 && r == 0){
      //  printf("%f \n", filteredImagesResized[0].at<float>(0,0) );
      //}
      cv::Mat margin_vector2 = cv::Mat::zeros(num_classes, 1, CV_32F);


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
        int k = 0;  //int offset=t*nTreeNodes, k=offset, k0=k;
        int k0 = k; //int val = (int)fids[offset];

        while((int)m_classifier["child"].at<float>(t,k)){
          int ftrId = static_cast<int>(m_classifier["fids"].at<float>(t,k));

          int ftrCids = cids[ftrId];
          int ftrChnIndex = (ftrCids+v)%nChan; //QUITAR LA V PARA COINCIDIR CON CHNS
          int ftrChnIndexPixelImg = (ftrCids+v)/nChan; //QUITAR LA V PARA COINCIDIR CON CHNS
          int ftrChnCol = ftrChnIndexPixelImg/(height*width);
          int ftrChnRow = ftrChnIndexPixelImg%(height*width);

          float ftr = filteredImagesResized[ftrChnIndex].at<float>(ftrChnCol , ftrChnRow); //float ftr = chns1[cids[(int)fids[k]]];
          float thrs = (m_classifier["thrs"].at<float>(t, k)); // thrs[k]

          int child_choosen = (ftr<thrs) ? 1 : 0; //k = (ftr<thrs[k]) ? 1 : 0;

          k = static_cast<int>(m_classifier["child"].at<float>(t, k0 )) - child_choosen ;  //float child = (m_classifier["child"].at<float>(t, k0)); // child[k0] ; k0 = k = child[k0]-k+offset;
          k0 = k ;
        }
        h = static_cast<int>((int)m_classifier["hs"].at<float>(t,k)); //h = static_cast<int>(hs[k]);
        

        cv::Mat Y2 = m_classifier["Y"](cv::Range(0,num_classes), cv::Range(h-1,h));
        cv::Mat update = m_classifier["w1_weights"].at<float>(t,0) * Y2;
        margin_vector2 += update;
        cv::Mat costs_vector =  m_classifier["Cprime"] * margin_vector2;
        double min_pos_cost1, max_pos_cost1; 
        cv::minMaxIdx(costs_vector.rowRange(1, num_classes), 
                      &min_pos_cost1, &max_pos_cost1);
        double neg_cost1 = costs_vector.at<float>(0,0); 
        

        trace = -(min_pos_cost1 - neg_cost1);

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
        cv::Rect2i rect;
        rect.x = cs[i] * stride;
        rect.y = rs[i] * stride;
        rect.width = modelWd;
        rect.height = modelHt;
        
        detections.push_back(rect);
    }
  }

  printf("size: %d\n", detections.size() );

  return detections;
}




























