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
#include <opencv2/opencv.hpp>
#include <channels/Utils.h>

#include <iostream>
#include <exception>

bool BadacostDetector::load
  (
  std::string clfPath,
  std::string pyrPath,
  std::string filtersPath)
{
  if (m_classifierIsLoaded)
  {
      return true;
  }

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
    //cv::transpose(m_Y, m_Y); // <----- JM: Cuidado la he traspuesto.
    
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

  cv::FileStorage pyramid;
  file_exists = pyramid.open(pyrPath, cv::FileStorage::READ);
  bool loadedOKPyr = false;
  if (file_exists)
  {
    std::cout << "Pyramids file exists!" << std::endl;

    // TODO: JM: Aquí hay que guardar la variable filters en el fichero yaml si es distinta de []. Si esa variable
    // no existe no se hace nada, si existe se pasan los filtros.
    loadedOKPyr = m_chnsPyramid.load(pyrPath.c_str());
  }

  cv::FileStorage filters;
  file_exists = pyramid.open(filtersPath, cv::FileStorage::READ);
  if (file_exists)
  {
    std::cout << "Filters exists!" << std::endl;
    m_filters = loadFilters(filtersPath);

    /*
    for (uint i=0; i < m_filters.size(); i++)
    {
      std::cout << m_filters[i] << std::endl;
    }
    */
    std::cout << "--> m_filters.size() = " << m_filters.size() << std::endl;
  }

  m_classifierIsLoaded = loadedOK && loadedOKPyr;
  return loadedOK;
}

std::vector<cv::Mat>
BadacostDetector::loadFilters(std::string filtersPath)
{
  //CARGAR EL FILTRO CREADO POR MATLAB DESDE UN YML
  cv::FileStorage filter;
  filter.open(filtersPath.c_str(), cv::FileStorage::READ);

  //OBTENER EL NOMBRE DE LOS DISTINTOS FILTROS PARA ESTE CASO
  std::vector<std::string> namesFilters;
  int num_filters_per_channel = 4; // <-- TODO: JM: Estos números tienen que venir en el fichero yaml!
  int num_channels = 10; // <-- TODO: JM: Estos números tienen que venir en el fichero yaml!
  int filter_size = 5; // <-- TODO: JM: Estos números tienen que venir en el fichero yaml!
  for(int i = 1; i <= num_filters_per_channel; i++)
  {
    for(int j = 1; j <= num_channels; j++)
    {
      std::string name  = "filter_" + std::to_string(j) + "_" + std::to_string(i);
      namesFilters.push_back(name);
    }
  }

  std::vector<cv::Mat> filters;
  for(uint k = 0; k < namesFilters.size(); k++)
  {
    cv::FileNode filterData = filter[namesFilters[k].c_str()]["data"];
    cv::FileNode filterRows = filter[namesFilters[k].c_str()]["rows"];
    cv::FileNode filterCols = filter[namesFilters[k].c_str()]["cols"];

    float* filt = new float[filter_size*filter_size*sizeof(float)];

    // TODO: Remove the need of this copy. It is because
    for(int i = 0; i < (int)filterRows; i++)
    {
      for(int j = 0; j < (int)filterCols; j++)
      {
        float x = (float)filterData[i*filter_size+j];
        filt[i*filter_size+j] = x;
      }
    }

    cv::Mat filterConver = cv::Mat(filter_size,filter_size, CV_32F, filt);
    transpose(filterConver,filterConver);
    filters.push_back(filterConver); //(filt);
  }

  return filters;
}

std::vector<cv::Rect2i>
BadacostDetector::detect(cv::Mat img)
{
  if (!m_classifierIsLoaded)
  {
    throw std::runtime_error("BadacostDetector::load() should be called befor BadacostDetector::detect()"); 
  }


  //CARGO LOS PARAMETROS, LLAMO A CHNSPYRAMID, SE PASA TODO POR EL FILTRO Y SE HACE RESIZE. 
  //EQUIVALENTE HASTA LINEA 80 acfDetectBadacost. Mismos resultados aparentemente.
  
  std::vector<cv::Mat> pyramid = m_chnsPyramid.getPyramid(img);
  std::vector<cv::Mat> filteredImagesResized;
  if (!m_filters.empty())
  {
    filteredImagesResized = m_chnsPyramid.badacostFilters(pyramid[0], m_filters);
  }
  else
  {
    filteredImagesResized = pyramid;
  }

/*
  for (int i=0; i < 10; i++)
  {
    cv::imshow("channel", filteredImagesResized[i*4]);
    cv::waitKey();
  }
*/
  printf("--> img's size = %d %d \n", img.size().height, img.size().width );
  printf("--> channel's size = %d %d \n", filteredImagesResized[0].size().height, filteredImagesResized[0].size().width );

 // cv::imshow("", filteredImagesResized[4]);
 // cv::waitKey(0);

  /*for(int i = 0; i < filteredImages.size(); i++)
  {
    cv::Mat imgResized = utils.ImgResample(filteredImages[i], 
                                           filteredImages[i].size().width/2, 
                                           filteredImages[i].size().height/2 );
    filteredImagesResized.push_back(imgResized);
  }*/

  // COMIENZA EL SEGUNDO BUCLE
  //int modelDsPad[2] = {54, 96}; // JM: Esto debería venir del fichero con el clasificador entrenado.
  //int modelDs[2] = {48, 84};    // JM: Esto debería venir del fichero con el clasificador entrenado.

  int shrink = 4;    // JM: Esto debería venir del fichero yaml con el clasificador entrenado.
  int modelHt = 54;  // JM: Esto debería venir del fichero yaml con el clasificador entrenado.
  int modelWd = 96;  // JM: Esto debería venir del fichero yaml con el clasificador entrenado.
  int stride = 2;    // JM: Esto debería venir del fichero yaml con el clasificador entrenado.
  float cascThr = -2.239887; // 1; // JM: Esto debería venir del fichero yaml con el clasificador entrenado.


  int height = filteredImagesResized[0].size().height;
  int width = filteredImagesResized[0].size().width;

  int nChan = filteredImagesResized.size();


  std::cout << "filteredImagesResized[0].size().height = " << filteredImagesResized[0].size().height << std::endl;
  std::cout << "filteredImagesResized[0].size().width = " << filteredImagesResized[0].size().width << std::endl;

  std::cout << "fids.size() = " << m_classifier["fids"].size << std::endl;
  std::cout << "fhild.size() = " << m_classifier["child"].size << std::endl;
  std::cout << "thrs.size() = " << m_classifier["thrs"].size << std::endl;
  std::cout << "hs.size() = " << m_classifier["hs"].size << std::endl;

  int nTreeNodes = m_classifier["fids"].size().width;
  int nTrees = m_classifier["fids"].size().height;
  std::cout << "nTrees = " << nTrees << std::endl;
  std::cout << "nTreeNodes = " << nTreeNodes << std::endl;
  int height1 = ceil(float((height*shrink)-modelHt+1)/stride);
  int width1 = ceil(float((width*shrink)-modelWd+1)/stride);

  int num_windows = width1*height1;
  if (num_windows < 0) 
  {
    // Detection window is too big for the image -> detect 0 windows, do nothing
    num_windows = 0;   
  }

  // These are needed for parallel processing of detection windows with OpenMP.
  // In any case it should works as is without OpenMP.
  std::vector<int> rs(num_windows, 0);
  std::vector<int> cs(num_windows, 0);
  std::vector<float> hs1(num_windows, 1.0); // Initialized to background class ("not object")
  std::vector<float> scores(num_windows, -1500.0); // Initialized to a very negative trace == score.

/*
  int nFtrs = (modelHt/shrink)*(modelWd/shrink)*nChan;
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
*/

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

  int modelWd_s = modelWd/shrink;
  int modelHt_s = modelHt/shrink;
  int modelWd_s_times_Ht_s = modelWd_s*modelHt_s;

  /*
    #ifdef USEOMP
    int nThreads = omp_get_max_threads();
    #pragma omp parallel for num_threads(nThreads)
    #endif
  */
  for( int c=0; c < width1; c++ )
  {
    //printf("%d %d \n", c, width1);
    for( int r=0; r < height1 ; r++ ) 
    { 
      //if(c == 0 && r == 0){
      //  printf("%f \n", filteredImagesResized[0].at<float>(0,0) );
      //}

      // Initialise the margin_vector memory to 0.0
      cv::Mat margin_vector = cv::Mat::zeros(m_num_classes, 1, CV_32F);
      float trace = 0.0;
      int h;

      //float *chns1=chns+(r*stride/shrink) + (c*stride/shrink)*height;
      int posHeight = (r*stride/shrink);
      int posWidth = (c*stride/shrink);      

      for(int t = 0; t < nTrees; t++ )
      {
        int k = 0; // Current tree node. We begin at the root (index 0).
        //int k0 = k;

        // La matriz "child" tiene los índices de los hijos almacenados por filas.
        float child_node_index = static_cast<int>(m_classifier["child"].at<float>(t,k));
        while( child_node_index ) // While k node is not a leave it has children (child_node_index != 0).
        {
          // std::cout << "t = " << t << ", k = " << k << ", child_node_index = " << child_node_index << std::endl;

          // Obtain the feature Id used in the split node.
          int ftrId = static_cast<int>(m_classifier["fids"].at<float>(t,k));

          // In the original code the m feature Id is redirected to this
          // cids value: 
          //    cids[m++] = z*width*height + c*height + r;    
          // Thus, we obtain the channel index z, column c and row for the feature from the ftrId:
          //   z - ftrChnIndex
          //   c - ftrChnCol
          //   r - ftrChnRow
          int ftrChnIndex = ftrId / (modelWd_s_times_Ht_s); // zsA[ftrId];
          int ftrChnCol = (ftrId % (modelWd_s_times_Ht_s)) / (modelHt_s); // csA[ftrId];
          ftrChnCol =  round(ftrChnCol + posWidth);
          int ftrChnRow = (ftrId % (modelWd_s_times_Ht_s)) % (modelHt_s); // rsA[ftrId]
          ftrChnRow =  round(ftrChnRow + posHeight);
/*
          std::cout << "=========" << std::endl;
          std::cout << "ftrId = " << ftrId << ", ";
          std::cout << "ftrChnIdex = " << ftrChnIndex << ", ";
          std::cout << "zsA[ftrId] = " << zsA[ftrId] << ", ";
          std::cout << "ftrChnCol = " << ftrChnCol << ", ";
          std::cout << "csA[ftrId]+posWidth = " <<  csA[ftrId]+posWidth << ", ";
          std::cout << "ftrChnRow = " << ftrChnRow << ", ";
          std::cout << "rsA[ftrId]+posHeight = " <<  rsA[ftrId]+posHeight << ", ";
*/

          // Obtain the feature value and threshold for the k-th tree node.
          float ftr = filteredImagesResized[ftrChnIndex].at<float>(ftrChnRow, ftrChnCol);
          float thrs = static_cast<float>(m_classifier["thrs"].at<float>(t, k));
          //printf("thrs %f  , ftr %f  \n", thrs, ftr);

          int child_choosen = (ftr<thrs) ? 1 : 0;

          // it seams the the right child of k0 is at k0 row index while 
          // left child is at k0-1 row index.
          //printf("child: %d, child_choosen: %d busquedaOffset %d \n", static_cast<int>(m_classifier["child"].at<float>(t,k0)), child_choosen, t*nTreeNodes);
          k = child_node_index  - child_choosen;
          child_node_index = static_cast<int>(m_classifier["child"].at<float>(t,k));
        }

        h = static_cast<int>(m_classifier["hs"].at<float>(t,k));
        //std::cout << "h = " << h << std::endl;

        // Add to the margin vector the codified output class h as a vector
        // multiplied by the weak learner weights     
        cv::Mat Y = m_Y(cv::Range(0,m_num_classes), cv::Range(h-1,h));

        //std::cout << "m_Y.size()=" << m_Y.size() << std::endl;
        //std::cout << "Y=" << Y << std::endl;
        //printf("--> %d \n", h);

        //printf("--> %f \n",(float)m_wl_weights.at<float>(0,0) );

        cv::Mat update = m_wl_weights.at<float>(0,t) * Y; //en la anterior estaba como (t,0) y daba resultado erroneo
        margin_vector += update;
        //std::cout << "margin_vector = " << margint_vector << std::endl;

        // Get the costs vector.
        cv::Mat costs_vector = m_Cprime * margin_vector;
        

        // Obtain the minimum cost for the positive classes (the object classes).
        // The negative class is the first and the rest are the positives
        double min_pos_cost;
        double max_pos_cost;
        cv::minMaxIdx(costs_vector.rowRange(1, m_num_classes), 
                      &min_pos_cost, &max_pos_cost);
        
        
        // Obtain the cost for the negative class (the background).
        // The negative class is the first and the rest are the positives
        float neg_cost = costs_vector.at<float>(0,0);
        //printf("%f %f \n", min_pos_cost, neg_cost);
        //printf("%f %f \n", min_pos_cost, neg_cost );

        // Get the trace for the current detection window
        trace = -(min_pos_cost - neg_cost);
        //std::cout << "trace = " << trace << std::endl;

        if (trace <= cascThr) break;
      }

      //printf("----------->%f \n",trace );
      if (trace < 0)
      {
        h = 1; // If trace is negative we have a background window (class is 1)
      }
  
      if (h > 1) // Otherwise we have a positive (object) class) 
      {
        int index = c + (r * width1);
        cs[index] = c; 
        rs[index] = r; 
        hs1[index] = h; 
        scores[index] = trace; 
      }
    }  
  } // End detector execution for all windows through rows and cols.

/*
  delete [] cids;
  delete [] zsA;
  delete [] csA;
  delete [] rsA;
*/

  // Obtain detection rectangles
  std::vector<cv::Rect2i> detections;
  std::vector<float> scores_out;
  std::vector<float> labels;

  for( uint i=0; i < hs1.size(); i++ )
  {
    if (hs1[i] > 1) // hs1[i]>1 are object windows, hs1[i]==1 are background windows.
    {
        cv::Rect2i rect;
        rect.x = cs[i] * stride;
        rect.y = rs[i] * stride;
        rect.width = modelWd;
        rect.height = modelHt;
        
        detections.push_back(rect);
        scores_out.push_back(scores[i]);
        labels.push_back(hs1[i]);
    }
  }

  std::cout <<  static_cast<uint>(detections.size()) << std::endl;
  return detections;
}




























