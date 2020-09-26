/** ------------------------------------------------------------------------
 *
 *  @brief badacostDetector.
 *  @author Jorge Vela
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2020/06/01
 *
 *  ------------------------------------------------------------------------ */


#include <detectors/badacostDetector.h>
#include <channels/ChannelsPyramid.h>
#include <opencv2/opencv.hpp>
#include <channels/Utils.h>

#include <iostream>
#include <exception>

#undef DEBUG
//#define DEBUG

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
      int rows = static_cast<int>(classifier[clf_variable_labels[i]]["cols"]); // <--- Cambiar en el scrip de guardado desde matlab (está al revés).
      int cols = static_cast<int>(classifier[clf_variable_labels[i]]["rows"]); // <--- Cambiar en el scrip de guardado desde matlab (está al revés).
      cv::FileNode data = classifier[clf_variable_labels[i]]["data"];
      
      cv::Mat matrix= cv::Mat::zeros(rows, cols, CV_32F);
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
    int rows = static_cast<int>(classifier["Cprime"]["cols"]); // <--- Cambiar en el scrip de guardado desde matlab (está al revés).
    int cols = static_cast<int>(classifier["Cprime"]["rows"]);
    cv::FileNode data = classifier["Cprime"]["data"];
    m_Cprime = cv::Mat::zeros(rows, cols, CV_32F);
    std::vector<float> p;
    data >> p;
    memcpy(m_Cprime.data, p.data(), p.size()*sizeof(float));
    transpose(m_Cprime, m_Cprime); // <-- We have transposed it!!

    // Read Y data
    rows = static_cast<int>(classifier["Y"]["cols"]); // <--- Cambiar en el scrip de guardado desde matlab (está al revés).
    cols = static_cast<int>(classifier["Y"]["rows"]); // <--- Cambiar en el scrip de guardado desde matlab (está al revés).
    data = classifier["Y"]["data"];
    m_Y = cv::Mat::zeros(rows, cols, CV_32F);
    data >> p;
    memcpy(m_Y.data, p.data(), p.size()*sizeof(float));
    
    // Read wl_weights data
    rows = static_cast<int>(classifier["w1_weights"]["cols"]); // <--- Cambiar en el scrip de guardado desde matlab (está al revés).
    cols = static_cast<int>(classifier["w1_weights"]["rows"]); // <--- Cambiar en el scrip de guardado desde matlab (está al revés).
    data = classifier["w1_weights"]["data"];
    m_wl_weights = cv::Mat::zeros(rows, cols, CV_32F);
    data >> p;
    memcpy(m_wl_weights.data, p.data(), p.size()*sizeof(float));
   

    // Read aRatio data
    rows = static_cast<int>(classifier["aRatio"]["cols"]); // <--- Cambiar en el scrip de guardado desde matlab (está al revés).
    cols = static_cast<int>(classifier["aRatio"]["rows"]); // <--- Cambiar en el scrip de guardado desde matlab (está al revés).
    data = classifier["aRatio"]["data"];
    m_aRatio = cv::Mat::zeros(rows, cols, CV_32F);
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
//  int filter_size = 5; // <-- TODO: JM: Estos números tienen que venir en el fichero yaml!
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
    cv::FileNode filterData = filter[namesFilters[k]]["data"];
    cv::FileNode filterRows = filter[namesFilters[k]]["cols"]; // <--- Cambiar en el scrip de guardado desde matlab (está al revés).
    cv::FileNode filterCols = filter[namesFilters[k]]["rows"]; // <--- Cambiar en el scrip de guardado desde matlab (está al revés).

    std::vector<float> p;
    cv::Mat filterConver = cv::Mat::zeros(filterRows, filterCols, CV_32F);
    filterData >> p;
    memcpy(filterConver.data, p.data(), p.size()*sizeof(float));
    transpose(filterConver,filterConver);

    // NOTE: filter2D is a correlation and to do convolution as in Matlab's conv2 we have to flip the kernels in advance.
    //       We have do it when loading them from file.
    cv::flip(filterConver, filterConver, -1);
    filters.push_back(filterConver);
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
  
  std::vector<std::vector<cv::Mat>> pyramid = m_chnsPyramid.compute(img, m_filters);
  std::vector<cv::Mat> filteredImagesResized;
  filteredImagesResized = pyramid[1];

//  if (!m_filters.empty())
//  {
//    filteredImagesResized = m_chnsPyramid.badacostFilters(pyramid[0], m_filters);
//  }
//  else
//  {
//    filteredImagesResized = pyramid;
//  }

#ifdef SHOW_CHANNELS
  for (int i=0; i < 40; i++)
  {
    std::cout << filteredImagesResized[i].size() << std::endl;
    cv::imshow("channel", filteredImagesResized[i]);
    cv::waitKey();
  }
#endif

#ifdef DEBUG
  std::cout << "--> img's size = " << img.size() << std::endl;
  std::cout << "--> channel's size = " << filteredImagesResized[0].size() << std::endl;
#endif

  // COMIENZA EL SEGUNDO BUCLE
  //cv::Size modelDsPad;
  //modelDsPad.width = 96; // JM: Esto debería venir del fichero con el clasificador entrenado.
  //modelDsPad.height = 54; // JM: Esto debería venir del fichero con el clasificador entrenado.
  //int modelDs[2] = {48, 84};    // JM: Esto debería venir del fichero con el clasificador entrenado.

  int shrink = 4;    // JM: Esto debería venir del fichero yaml con el clasificador entrenado.
  int modelHt = 54;  // JM: Esto debería venir del fichero yaml con el clasificador entrenado.
  int modelWd = 96;  // JM: Esto debería venir del fichero yaml con el clasificador entrenado.
  int stride = 4;    // JM: Esto debería venir del fichero yaml con el clasificador entrenado.
  float cascThr = -2.239887; // 1; // JM: Esto debería venir del fichero yaml con el clasificador entrenado.


  int height = filteredImagesResized[0].size().height;
  int width = filteredImagesResized[0].size().width;


#ifdef DEBUG
  std::cout << "filteredImagesResized[0].size().height = " << filteredImagesResized[0].size().height << std::endl;
  std::cout << "filteredImagesResized[0].size().width = " << filteredImagesResized[0].size().width << std::endl;

  std::cout << "fids.size() = " << m_classifier["fids"].size() << std::endl;
  std::cout << "fhild.size() = " << m_classifier["child"].size() << std::endl;
  std::cout << "thrs.size() = " << m_classifier["thrs"].size() << std::endl;
  std::cout << "hs.size() = " << m_classifier["hs"].size() << std::endl;
  std::cout << "hs.size().width = " << m_classifier["hs"].size().width << std::endl;
  std::cout << "hs.size().height = " << m_classifier["hs"].size().height << std::endl;
#endif

  int nTreeNodes = m_classifier["fids"].size().width;
  int nTrees = m_classifier["fids"].size().height;
  int height1 = ceil(float((height*shrink)-modelHt+1)/stride);
  int width1 = ceil(float((width*shrink)-modelWd+1)/stride);

  std::cout << "nTrees = " << nTrees << std::endl;
  std::cout << "nTreeNodes = " << nTreeNodes << std::endl;
  std::cout << "height1 = " << height1 << std::endl;
  std::cout << "width1 = " << width1 << std::endl;

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
        //cids[m++] = z*width*height + c*height + r;
      }
*/

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
    for( int r=0; r < height1 ; r++ ) 
    { 
      //std::cout << "c = " << c << ", " << "r = " << r << ", " << "width1 = " << width1 << ", " << "height1 = " << height1 << std::endl;

      // Initialise the margin_vector memory to 0.0
      cv::Mat margin_vector = cv::Mat::zeros(m_num_classes, 1, CV_32F);
      cv::Mat costs_vector = cv::Mat::zeros(m_num_classes, 1, CV_32F);
      float trace = 0.0;
      int h;

      //float *chns1=chns+(r*stride/shrink) + (c*stride/shrink)*height;
      int posHeight = (r*stride/shrink);
      int posWidth = (c*stride/shrink);      

      int t;
      for(t = 0; t < nTrees; t++ )
      {
        int k = 0; // Current tree node. We begin at the root (index 0).

        // La matriz "child" tiene los índices de los hijos almacenados por filas.
        float child_node_index = static_cast<int>(m_classifier["child"].at<float>(t, k));
        float ftr;
        float thrs;
        int ftrId;
        while( child_node_index ) // While k node is not a leave it has children (child_node_index != 0).
        {

          //printf("child : %d \n", (int)m_classifier["child"].at<float>(t,k));
          // Obtain the feature Id used in the split node.
          ftrId = static_cast<int>(m_classifier["fids"].at<float>(t, k));

          // In the original code the m feature Id is redirected to this
          // cids value: 
          //    cids[m++] = z*width*height + c*height + r;    
          // Thus, we obtain the channel index z, column c and row for the feature from the ftrId:
          //   z - ftrChnIndex
          //   c - ftrChnCol
          //   r - ftrChnRow
          int ftrChnIndex = ftrId / (modelWd_s_times_Ht_s); // zsA[ftrId]
          int ftrChnCol = (ftrId % (modelWd_s_times_Ht_s)) / (modelHt_s); // csA[ftrId]
          ftrChnCol =  round(ftrChnCol + posWidth);
          int ftrChnRow = (ftrId % (modelWd_s_times_Ht_s)) % (modelHt_s); // rsA[ftrId]
          ftrChnRow =  round(ftrChnRow + posHeight);

/*
          std::cout << "ftrId = " << ftrId << ", ";
          std::cout << "ftrChnIdex = " << ftrChnIndex << ", ";
          std::cout << "zsA[ftrId] = " << zsA[ftrId] << ", ";
          std::cout << "ftrChnCol = " << ftrChnCol << ", ";
          std::cout << "csA[ftrId]+posWidth = " <<  csA[ftrId]+posWidth << ", ";
          std::cout << "ftrChnRow = " << ftrChnRow << ", ";
          std::cout << "rsA[ftrId]+posHeight = " <<  rsA[ftrId]+posHeight << ", ";
*/

          // Obtain the feature value and threshold for the k-th tree node.
          ftr = filteredImagesResized[ftrChnIndex].at<float>(ftrChnRow, ftrChnCol);
          thrs = static_cast<float>(m_classifier["thrs"].at<float>(t, k));

#ifdef DEBUG
          if (t==0)
          {
            std::cout << "***" << "k = " << k << ", " << "ftr = " << ftr << ", " << "thrs = " << thrs << ", " << "fids[k] = " << ftrId;
            std::cout << ", ftrChnIndex= " << ftrChnIndex;
            std::cout << ", ftrChnRow= " << ftrChnRow;
            std::cout << ", ftrChnCol= " << ftrChnCol << std::endl;
          }
#endif
          int child_choosen = (ftr<thrs) ? 1 : 0;                   

          // The right child of k is at child_node_index while
          // left child is at child_node_index-1 index.
          k = child_node_index  - child_choosen;
          child_node_index = static_cast<int>(m_classifier["child"].at<float>(t, k));
        }
        h = static_cast<int>(m_classifier["hs"].at<float>(t, k));

        // Add to the margin vector the codified output class h as a vector
        // multiplied by the weak learner weights     
        cv::Mat Y = m_Y(cv::Range(0,m_num_classes), cv::Range(h-1,h));
        cv::Mat update = m_wl_weights.at<float>(0, t) * Y; // Here m_wl_weights is a row vector, thus (0, t) is the right access.
        margin_vector  = margin_vector + update;

        // Get the costs vector.
        costs_vector = m_Cprime * margin_vector;

        // Obtain the minimum cost for the positive classes (the object classes).
        // The negative class is the first and the rest are the positives
        double min_positive_cost;
        double max_positive_cost;
        cv::minMaxIdx(costs_vector.rowRange(1, m_num_classes), 
                      &min_positive_cost, &max_positive_cost);

        // Obtain the cost for the negative class (the background).
        // The negative class is the first and the rest are the positives
        float neg_cost = costs_vector.at<float>(0,0);

        // Get the trace for the current detection window
        trace = -(min_positive_cost - neg_cost);

        if (trace <= cascThr) break;
      }

#ifdef DEBUG
      std::cout << "c = " << c << ", " << "r = " << r << ", " << "t = " << t << ", " << "h = " << h << std::endl;
#endif

      if (trace < 0)
      {
        h = 1;
      }
      else //if (trace > 0)
      //if (h > 1)
      {
        // WARNING: Change with respect to the Matlab's implementation ... there is a bug in the matlab implementation
        //          as it returns the h (positive class) of the last executed tree and not the minimum cost one !!! :-(.
        // If trace is negative we have a background window (class is 1)
        // Otherwise we have a positive (object) class)
//        double min_cost;
//        double max_cost;
//        int min_ind[2];
//        int max_ind[2];
//        cv::minMaxIdx(costs_vector.rowRange(1,m_num_classes),
//                      &min_cost, &max_cost, min_ind, max_ind, cv::Mat());
//        h = min_ind[0] + 2; // +1 because of 0 index, and +1 because of negative class is 1.
        // End of corrected code w.r.t. Matlab's implementation.

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

  return detections;
}




























