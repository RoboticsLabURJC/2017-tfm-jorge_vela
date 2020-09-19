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
#include <exception>

bool BadacostDetector::load(std::string clfPath)
{
  bool loadedOK = false;

  cv::FileStorage classifier;
  std::map<std::string, cv::Mat> clf;
  bool file_exists = classifier.open(clfPath, cv::FileStorage::READ);

  if (file_exists)
  {
    std::string clf_variable_labels[14] = {"fids", "thrs", "child", "hs", 
      "weights", "depth"};
   
    // Remove any existing variables from the classifier map.
    m_classifier.clear();
      
    for(int i = 0; i < 14; i++)
    {
      int rows = static_cast<int>(classifier[clf_variable_labels[i]]["rows"]);
      int cols = static_cast<int>(classifier[clf_variable_labels[i]]["cols"]);
      cv::FileNode data = classifier[clf_variable_labels[i]]["data"];
      
      cv::Mat matrix= cv::Mat::zeros(cols, rows, CV_32F);
      std::vector<float> p;
      data >> p;
      memcpy(matrix.data, p.data(), p.size()*sizeof(float));

      m_classifier.insert({clf_variable_labels[i].c_str(), matrix });    
    }
    

    cv::FileNode dataNumClases = classifier["num_classes"]["data"];
    cv::Mat m_num_clss = cv::Mat::zeros(1, 1, CV_32F);
    std::vector<float> pNumClasses;
    dataNumClases >> pNumClasses;
    memcpy(m_num_clss.data, pNumClasses.data(), pNumClasses.size()*sizeof(float));
    m_num_classes = (int)m_num_clss.at<float>(0,0);


    cv::FileNode dataTreeDepth= classifier["treeDepth"]["data"];
    cv::Mat m_tr_dpth = cv::Mat::zeros(1, 1, CV_32F);
    std::vector<float> pTrDpt;
    dataTreeDepth >> pTrDpt;
    memcpy(m_tr_dpth.data, pTrDpt.data(), pTrDpt.size()*sizeof(float));
    m_treeDepth = (int)m_tr_dpth.at<float>(0,0);


    cv::FileNode dataRatioFixed= classifier["aRatioFixedWidth"]["data"];
    cv::Mat m_RatioFixed = cv::Mat::zeros(1, 1, CV_32F);
    std::vector<float> pRdFix;
    dataRatioFixed >> pRdFix;
    memcpy(m_RatioFixed.data, pRdFix.data(), pRdFix.size()*sizeof(float));
    m_aRatioFixedWidth = (int)m_RatioFixed.at<float>(0,0);

    //m_treeDepth = classifier["treeDepth"]["data"];    
    //m_num_classes = classifier["num_classes"]["data"];   
    //m_aRatioFixedWidth = classifier["aRatioFixedWidth"]["data"];    
    

    // Read Cprime data
    int rows = static_cast<int>(classifier["Cprime"]["rows"]);
    int cols = static_cast<int>(classifier["Cprime"]["cols"]);
    cv::FileNode data = classifier["Cprime"]["data"];
    m_Cprime = cv::Mat::zeros(cols, rows, CV_32F);
    std::vector<float> p;
    data >> p;
    memcpy(m_Cprime.data, p.data(), p.size()*sizeof(float));


    // Read Y data
    rows = static_cast<int>(classifier["Y"]["rows"]);
    cols = static_cast<int>(classifier["Y"]["cols"]);
    data = classifier["Y"]["data"];
    m_Y = cv::Mat::zeros(cols, rows, CV_32F);
    data >> p;
    memcpy(m_Y.data, p.data(), p.size()*sizeof(float));

    
    // Read wl_weights data
    rows = static_cast<int>(classifier["w1_weights"]["rows"]);
    cols = static_cast<int>(classifier["w1_weights"]["cols"]);
    data = classifier["w1_weights"]["data"];
    m_wl_weights = cv::Mat::zeros(cols, rows, CV_32F);
    data >> p;
    memcpy(m_wl_weights.data, p.data(), p.size()*sizeof(float));
   

    // Read aRatio data
    rows = static_cast<int>(classifier["aRatio"]["rows"]);
    cols = static_cast<int>(classifier["aRatio"]["cols"]);
    data = classifier["aRatio"]["data"];
    m_aRatio = cv::Mat::zeros(cols, rows, CV_32F);
    data >> p;
    memcpy(m_aRatio.data, p.data(), p.size()*sizeof(float));

    loadedOK = true;
  }
  m_classifierIsLoaded = loadedOK;
  return loadedOK;
}

std::vector<cv::Rect2i>
BadacostDetector::detect(cv::Mat imgs)
{
  Utils utils;
  ChannelsPyramid chnsPyramid;

  if (!m_classifierIsLoaded)
  {
    throw std::runtime_error("BadacostDetector::load() should be called befor BadacostDetector::detect()"); 
  }

  //CARGO LOS PARAMETROS, LLAMO A CHNSPYRAMID, SE PASA TODO POR EL FILTRO Y SE HACE RESIZE. 
  //EQUIVALENTE HASTA LINEA 80 acfDetectBadacost. Mismos resultados aparentemente.
  
  // JM: Aquí hay que guardar la variable filters en el fichero yaml si es distinta de []. Si esa variable
  // no existe no se hace nada, si existe se pasan los filtros.
  std::string nameOpts = "yaml/pPyramid.yml";
  bool loadOk = chnsPyramid.load(nameOpts.c_str());
  std::vector<cv::Mat> pyramid = chnsPyramid.getPyramid(imgs);  


  //std::vector<cv::Mat> filteredImages = chnsPyramid.badacostFilters(pyramid[0], "yaml/filterTest.yml");

  std::vector<cv::Mat> filteredImagesResized= chnsPyramid.badacostFilters(pyramid[0], "yaml/filterTest.yml");


  printf("--> %d %d \n", filteredImagesResized[0].size().height, filteredImagesResized[0].size().width );

  cv::imshow("", filteredImagesResized[4]);
  cv::waitKey(0);

  /*for(int i = 0; i < filteredImages.size(); i++)
  {
    cv::Mat imgResized = utils.ImgResample(filteredImages[i], 
                                           filteredImages[i].size().width/2, 
                                           filteredImages[i].size().height/2 );
    filteredImagesResized.push_back(imgResized);
  }*/

  // COMIENZA EL SEGUNDO BUCLE
  int modelDsPad[2] = {54, 96}; // JM: Esto debería venir del fichero con el clasificador entrenado.
  int modelDs[2] = {48, 84};    // JM: Esto debería venir del fichero con el clasificador entrenado.

  int shrink = 4;    // JM: Esto debería venir del fichero yaml con el clasificador entrenado.
  int modelHt = 54;  // JM: Esto debería venir del fichero yaml con el clasificador entrenado.
  int modelWd = 96;  // JM: Esto debería venir del fichero yaml con el clasificador entrenado.
  int stride = 4;    // JM: Esto debería venir del fichero yaml con el clasificador entrenado.
  float cascThr = -2.239887; // 1; // JM: Esto debería venir del fichero yaml con el clasificador entrenado.


  int height = filteredImagesResized[0].size().height;
  int width = filteredImagesResized[0].size().width;
  int nChan = filteredImagesResized.size();

  int nTreeNodes = m_classifier["fids"].size().width;
  int nTrees = m_classifier["fids"].size().height;
  int height1 = ceil(float(height*shrink-modelHt+1)/stride);
  int width1 = ceil(float((width*shrink)-modelWd+1)/stride);
  int nFtrs = (modelHt/shrink)*(modelWd/shrink)*nChan;
    

  // JM: Esto debe de ser así siempre y por eso hacemos un assert:
  //assert((modelHt/shrink) == height);
  //assert((modelWd/shrink) == width);
  
  int num_windows = width1*height1;
  if (num_windows < 0) 
  {
    // Detection window is too big for the image -> detect 0 windows, do nothing
    num_windows = 0;   
  }

  // These are needed for parallel processing of detection windows with OpenMP.
  // In any case it should works as is without OpenMP.
  std::vector<int> rs(num_windows), cs(num_windows); 
  std::vector<float> hs1(num_windows), scores(num_windows);

  int *cids =  new int[nFtrs];
  int *zsA = new int[nFtrs];
  int *csA = new int[nFtrs];
  int *rsA = new int[nFtrs];
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

    /*int t = 0;
    for (int k = 0; k < 177; k++){ //k=1:177
        for (int v = 0; v < 313; v++){ //v=1:313
          for (int w = 0; w < 40; w++){ //w=1:40
              if(t %2 == 0){
              filteredImagesResized[w].at<float>(k,v) = -t ; 
              }else{
                filteredImagesResized[w].at<float>(k,v) = t ; 
              }
              t = t + 7.0;
               //printf("%d %d %d \n",k,v,w );
            }
        }
    }*/

/*  
  #ifdef USEOMP
  int nThreads = omp_get_max_threads();
  #pragma omp parallel for num_threads(nThreads)
  #endif
*/
  for( int c=0; c< width1; c++ )
  {

    for( int r=0; r< height1 ; r++ ) 
    { 


      //if(c == 0 && r == 0){
      //  printf("%f \n", filteredImagesResized[0].at<float>(0,0) );
      //}

      // Initialise the margin_vector memory to 0.0
      cv::Mat margin_vector = cv::Mat::zeros(m_num_classes, 1, CV_32F);
      double trace;
      int h;

      //float *chns1=chns+(r*stride/shrink) + (c*stride/shrink)*height;
      int posHeight = (r*stride/shrink);
      int posWidth = (c*stride/shrink);
      
      for(int t = 0; t < nTrees; t++ ) 
      {
        int k = 0; // Empezamos sobre el primer nodo del t-ésimo árbol.
        int k0 = k; 
        // La matriz "child" tiene los árboles almacenados por columnas.
        while((int)m_classifier["child"].at<float>(t,k))
        {
          // Obtain the feature Id used in the split node.
          int ftrId = static_cast<int>(m_classifier["fids"].at<float>(t,k));
 

          int chanSearch = zsA[ftrId];
          int hSearch = csA[ftrId];
          int wSearch = rsA[ftrId];
          //printf("%d %d %d \n",chanSearch, hSearch, wSearch );
          float ftr2 = (float)filteredImagesResized[chanSearch].at<float>(wSearch+(r*stride/shrink), hSearch + (c*stride/shrink));      
          //printf("%f \n", ftr2);   
          // In the original code the m feature Id is redirected to this 
          // cids value: 
          //    cids[m++] = z*width*height + c*height + r;    
          // Thus, we obtain the channel index z, column c and row for the feature:
          //   z - ftrChnIndex
          //   c - ftrChnCol
          //   r - ftrChnRow
          int ftrChnIndex = ftrId / (width*height);
          int ftrChnCol = (ftrId % (width*height)) / height;
          int ftrChnRow =  (ftrId % (width*height)) % height;

          // Obtain the feature value and threshold for the k-th tree node.
          //float ftr = filteredImagesResized[ftrChnIndex].at<float>(ftrChnRow, ftrChnCol);
          //printf("%f \n", ftr2 );
          float thrs = static_cast<float>(m_classifier["thrs"].at<float>(t, k));
          //printf("thrs %f  , ftr %f  \n", thrs, ftr2);

          int child_choosen = (ftr2<thrs) ? 1 : 0;

          // it seams the the right child of k0 is at k0 row index while 
          // left child is at k0-1 row index.
          //printf("child: %d, child_choosen: %d busquedaOffset %d \n", static_cast<int>(m_classifier["child"].at<float>(t,k0)), child_choosen, t*nTreeNodes);

          k = static_cast<int>(m_classifier["child"].at<float>(t,k0)) - child_choosen;
          k0 = k;

        }
        h = static_cast<int>((int)m_classifier["hs"].at<float>(t,k));
        // Add to the margin vector the codified output class h as a vector 
        // multiplied by the weak learner weights     
        cv::Mat Y = m_Y(cv::Range(0,m_num_classes), cv::Range(h-1,h));
        //printf("--> %d \n", h);


        //for(int wk = 0; wk < Y.size().height; wk++)
        //  printf("%f \n", (float)Y.at<float>(wk,0) );

        //printf("--> %f \n",(float)m_wl_weights.at<float>(0,0) );
        cv::Mat update = m_wl_weights.at<float>(t,0) * Y;
        margin_vector += update;

        // Get the costs vector.
        cv::Mat costs_vector = m_Cprime * margin_vector;
        
        // Obtain the minimum cost for the positive classes (the object classes).
        // The negative class is the first and the rest are the positives
        double min_pos_cost, max_pos_cost; 
        cv::minMaxIdx(costs_vector.rowRange(1, m_num_classes), 
                      &min_pos_cost, &max_pos_cost);
        
        
       // Obtain the cost for the negative class (the background).
        // The negative class is the first and the rest are the positives
        double neg_cost = costs_vector.at<float>(0,0); 
        
        //printf("%f %f \n", min_pos_cost, neg_cost );
        // Get the trace for the current detection window
        trace = -(min_pos_cost - neg_cost);
        //printf("%f \n",trace );
        if (trace <= cascThr) break;
      }

      if (trace < 0)
      {
        h=1; // If trace is negative we have a background window (class is 1)
      }
  
      if (h > 1) // Otherwise we have a positive (object) class) 
      {
        int index = c + (r*width1);
        cs[index] = c; 
        rs[index] = r; 
        hs1[index] = h; 
        scores[index] = trace; 
      }
    }  
  } // End detector execution for all windows through rows and cols.
  

  // Obtain detection rectangles
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
  printf("%d \n", detections.size());
  return detections;
}




























