/** ------------------------------------------------------------------------
 *
 *  @brief badacostDetector.
 *  @author Jorge Vela
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2020/06/01
 *
 *  ------------------------------------------------------------------------ */


#include <detectors/BadacostDetector.h>
#include <pyramid/ChannelsPyramid.h>
#include <opencv2/opencv.hpp>
#include <channels/Utils.h>

#include <iostream>
#include <exception>

#undef DEBUG
//#define DEBUG

#undef SHOW_CHANNELS
//#define SHOW_CHANNELS

BadacostDetector::BadacostDetector
  (
  std::string channels_pyramid_impl,
  std::string channels_impl,
  float minScore
  )
  {
    m_classifierIsLoaded = false;
    m_channels_pyramid_impl = channels_pyramid_impl;
    m_channels_impl = channels_impl;
    if (m_channels_pyramid_impl != "opencl")
    {
      m_pChnsPyramidStrategy = ChannelsPyramid::createChannelsPyramid(channels_pyramid_impl, channels_impl);
    }
    m_minScore = minScore;
  };

BadacostDetector::~BadacostDetector
  ()
{}

bool BadacostDetector::load
  (
  std::string clfPath,
  std::string filtersPath
  )
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
    std::string clf_variable_labels[14] = {"fids", "thrs", "child", "hs", "weights", "depth"};
   
    // Remove any existing variables from the classifier map.
    m_classifier.clear();
    std::vector<float> p;
    for(int i = 0; i < 14; i++)
    {
      cv::Mat matrix = readMatrixFromFileNode(classifier[clf_variable_labels[i]]);
      m_classifier.insert({clf_variable_labels[i].c_str(), matrix });    
    }  
    m_num_classes = static_cast<int>(readScalarFromFileNode(classifier["num_classes"]));
    m_treeDepth = static_cast<int>(readScalarFromFileNode(classifier["treeDepth"]));
    m_aRatioFixedWidth = static_cast<bool>(readScalarFromFileNode(classifier["aRatioFixedWidth"]));

    // Read Cprime data
    m_Cprime = readMatrixFromFileNode(classifier["Cprime"]);
    // Read Y data
    m_Y = readMatrixFromFileNode(classifier["Y"]);
    // Read wl_weights data
    m_wl_weights = readMatrixFromFileNode(classifier["w1_weights"]);
   
    // Read aRatio data
    if (!classifier["aRatio"].empty())
    {
      m_aRatio = readMatrixFromFileNode(classifier["aRatio"]);
    }

    loadedOK = true;
  }

  ClassifierConfig clfData;

  clfData.modelDs.width = classifier["modelDs"]["data"][1];
  clfData.modelDs.height = classifier["modelDs"]["data"][0];

  clfData.modelDsPad.width = classifier["modelDsPad"]["data"][1];
  clfData.modelDsPad.height = classifier["modelDsPad"]["data"][0];

  clfData.stride = classifier["stride"]["data"][0];
  clfData.cascThr = float(classifier["cascThr"]["data"][0]);

  clfData.luv.smooth = classifier["pChns.pColor"]["smooth"];

  clfData.luv.smooth_kernel_size = 1;

  clfData.padding.width = classifier["pad"]["data"][1];
  clfData.padding.height = classifier["pad"]["data"][0];
  clfData.nOctUp = classifier["nOctUp"]["data"][0];
  clfData.nPerOct = classifier["nPerOct"]["data"][0];
  clfData.nApprox = classifier["nApprox"]["data"][0];
  clfData.shrink = classifier["pChns.shrink"]["data"];

  std::cout << "clfData.nOctUp = " << clfData.nOctUp << std::endl;
  std::cout << "clfData.nPerOct = " << clfData.nPerOct << std::endl;
  std::cout << "clfData.nApprox = " << clfData.nApprox << std::endl;
  std::cout << "clfData.shrink = " << clfData.shrink << std::endl;

  m_shrink = clfData.shrink*2; // JM: only x2 with LDCF features!!!

  clfData.gradMag.normRad = classifier["pChns.pGradMag"]["normRad"]; 
  clfData.gradMag.normConst = classifier["pChns.pGradMag"]["normConst"]; 

  clfData.gradHist.binSize = classifier["pChns.pGradHist"]["enabled"];
  clfData.gradHist.nOrients = classifier["pChns.pGradHist"]["nOrients"];
  clfData.gradHist.softBin = classifier["pChns.pGradHist"]["softBin"];
  clfData.gradHist.full = false;


  int lambdasSize = classifier["lambdas"]["rows"];
  for(int i = 0; i < lambdasSize; i++)
  {
    clfData.lambdas.push_back((float)classifier["lambdas"]["data"][i]);
  }

  clfData.minDs.width = classifier["minDs"]["data"][1]; 
  clfData.minDs.height = classifier["minDs"]["data"][0]; 

  m_clfData = clfData;

  cv::FileStorage filters;
  file_exists = filters.open(filtersPath, cv::FileStorage::READ);
  if (file_exists)
  {
    m_filters = loadFilters(filtersPath);
  }

  m_classifierIsLoaded = loadedOK; // && loadedOKPyr;
  return loadedOK;
}

std::vector<cv::Mat>
BadacostDetector::loadFilters(std::string filtersPath)
{
  // Load the LDCF filters
  cv::FileStorage filter;
  filter.open(filtersPath.c_str(), cv::FileStorage::READ);

  // Obtain the LDCF filters names
  std::vector<std::string> namesFilters;
  int num_filters_per_channel = 4; // <-- TODO: JM: Estos números tienen que venir en el fichero yaml!
  int num_channels = 10; // <-- TODO: JM: Estos números tienen que venir en el fichero yaml!
  for(int i = 1; i <= num_filters_per_channel; i++)
  {
    for(int j = 1; j <= num_channels; j++)
    {
      std::string name  = "filter_" + std::to_string(j) + "_" + std::to_string(i);
      namesFilters.push_back(name);
    }
  }

  std::vector<cv::Mat> filters;
  std::vector<float> p;
  for(uint k = 0; k < namesFilters.size(); k++)
  {
    cv::FileNode filterData = filter[namesFilters[k]]["data"];
    cv::FileNode filterRows = filter[namesFilters[k]]["cols"]; // <--- Cambiar en el scrip de guardado desde matlab (está al revés).
    cv::FileNode filterCols = filter[namesFilters[k]]["rows"]; // <--- Cambiar en el scrip de guardado desde matlab (está al revés).

    cv::Mat filterConver = cv::Mat::zeros(filterRows, filterCols, CV_32F);
    p.clear();
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

void BadacostDetector::correctToClassSpecificBbs
  (
  std::vector<DetectionRectangle>& dts,
  std::vector<float> aRatios,
  bool fixedWidth // In this case we keep the h fixed and modify w
  )
{
  // We assume that the background class has index 1. So we have to
  // remove 1 to the positive class index to get the correct median
  // aspect ratio. Therefore, if the positive class label is i, we get
  // its aspect ratio for its BBoxes as aRatio(i-1).

  if (dts.size() == 0)
  {
    return;
  }

  int squarify_param = 0;
  if (fixedWidth)
  {
    squarify_param = 2; // use original w, alter h
  }
  else
  {
    squarify_param = 3; // use original h, alter w
  }

  for (uint i = 0; i < dts.size(); i++)
  {
    dts[i].squarify(squarify_param, aRatios[dts[i].class_index-2]);
  }
}

std::vector<DetectionRectangle>
BadacostDetector::detect(cv::Mat img)
{
  if (!m_classifierIsLoaded)
  {
    throw std::runtime_error("BadacostDetector::load() should be called befor BadacostDetector::detect()");
  }

  // Compute feature channels pyramid
  std::vector<std::vector<cv::Mat>> pyramid;
  std::vector<double> scales;
  std::vector<cv::Size2d> scaleshw;

  if (m_channels_pyramid_impl != "opencl")
  {
    pyramid = m_pChnsPyramidStrategy->compute(img, m_filters, scales, scaleshw, m_clfData);
  }
  else
  {
    // CPU->GPU
    cv::UMat img_gpu = cv::UMat(img.size(), img.type(), cv::USAGE_ALLOCATE_DEVICE_MEMORY);
    img.copyTo(img_gpu);

    // Computation in GPU and GPU->CPU
    pyramid = m_chnsPyramidOpenCL.compute(img_gpu, m_filters, scales, scaleshw, m_clfData);
  }

  // Execute the detector over all the scales
  std::vector<DetectionRectangle> detections;
  for (uint i = 0; i < pyramid.size(); i++)
  {
    std::vector<DetectionRectangle> detections_i = detectSingleScale(pyramid[i]);

    int shift_x = round((m_clfData.modelDsPad.width - m_clfData.modelDs.width)/2.0) - m_clfData.padding.width;
    int shift_y = round((m_clfData.modelDsPad.height - m_clfData.modelDs.height)/2.0) - m_clfData.padding.height;
    for (uint j = 0; j < detections_i.size(); j++)
    {

      DetectionRectangle d = detections_i[j];
      d.bbox.x = (d.bbox.x + shift_x) / scaleshw[i].width;
      d.bbox.y = (d.bbox.y + shift_y) / scaleshw[i].height;
      d.bbox.width = m_clfData.modelDs.width / scales[i];
      d.bbox.height = m_clfData.modelDs.height / scales[i];

      detections.push_back(d);
    }
  }

  if  (m_aRatio.rows > 0)
  {
    // Change bounding boxes to class specific bounding box. For
    // example in the KITTI benchmark we have cars. At each orientation the
    // bounding box of the car has a specific aspect ratio (a size view of the
    // car is rectangular and a frontal car is squared).
    correctToClassSpecificBbs(detections, m_aRatio, m_aRatioFixedWidth);
  }

  std::vector<DetectionRectangle> detections_nms;
  nonMaximumSuppression(detections, detections_nms);

#ifdef DEBUG
  std::cout << "detections.size() = " << detections.size() << std::endl;
  std::cout << "detections_nms.size() = " << detections_nms.size() << std::endl;
#endif

  return detections_nms;
}

std::vector<DetectionRectangle>
BadacostDetector::detectSingleScale
  (
  std::vector<cv::Mat>& channels
  )
{

#ifdef SHOW_CHANNELS
  for (int i=0; i < 40; i++)
  {
    std::cout << channels[i].size() << std::endl;
    cv::imshow("channel", channels[i]);
    cv::waitKey();
  }
#endif

#ifdef DEBUG
  std::cout << "--> channel's size = " << channels[0].size() << std::endl;
#endif

  int height = channels[0].size().height;
  int width = channels[0].size().width;


#ifdef DEBUG
  std::cout << "channels[0].size().height = " << channels[0].size().height << std::endl;
  std::cout << "channels[0].size().width = " << channels[0].size().width << std::endl;

  std::cout << "fids.size() = " << m_classifier["fids"].size() << std::endl;
  std::cout << "fhild.size() = " << m_classifier["child"].size() << std::endl;
  std::cout << "thrs.size() = " << m_classifier["thrs"].size() << std::endl;
  std::cout << "hs.size() = " << m_classifier["hs"].size() << std::endl;
  std::cout << "hs.size().width = " << m_classifier["hs"].size().width << std::endl;
  std::cout << "hs.size().height = " << m_classifier["hs"].size().height << std::endl;
#endif

  int nTrees = m_classifier["fids"].size().height;
  int height1 = ceil(float((height*m_shrink)-m_clfData.modelDsPad.height+1)/m_clfData.stride);
  int width1 = ceil(float((width*m_shrink)-m_clfData.modelDsPad.width+1)/m_clfData.stride);

#ifdef DEBUG
  std::cout << "nTrees = " << nTrees << std::endl;
  std::cout << "height1 = " << height1 << std::endl;
  std::cout << "width1 = " << width1 << std::endl;
#endif

  int num_windows = width1*height1;
  if (num_windows < 0) 
  {
    // Detection window is too big for the image -> detect 0 windows, do nothing
    num_windows = 0;   
  }

  // These are needed for parallel processing of detection windows.
  // In any case it should works as is without a parallel execution.
  std::vector<int> rs(num_windows, 0);
  std::vector<int> cs(num_windows, 0);
  std::vector<float> hs1(num_windows, 1.0); // Initialized to background class ("not object")
  std::vector<float> scores(num_windows, -1500.0); // Initialized to a very negative trace == score.

  int modelWd_s = m_clfData.modelDsPad.width/m_shrink;
  int modelHt_s = m_clfData.modelDsPad.height/m_shrink;
  int modelWd_s_times_Ht_s = modelWd_s*modelHt_s;

  cv::parallel_for_(cv::Range(0, width1*height1), [&](const cv::Range& rang)
  {
    for (int k = rang.start; k < rang.end; k++)
    {
      int r = k / width1;
      int c = k % width1;

      // std::cout << "c = " << c << ", " << "r = " << r << ", " << "width1 = " << width1 << ", " << "height1 = " << height1 << std::endl;

      // Initialise the margin_vector memory to 0.0
      cv::Mat margin_vector = cv::Mat::zeros(m_num_classes, 1, CV_32F);
      cv::Mat costs_vector = cv::Mat::zeros(m_num_classes, 1, CV_32F);
      float trace = 0.0;
      int h;
      int posHeight = (r*m_clfData.stride/m_shrink);
      int posWidth = (c*m_clfData.stride/m_shrink);

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
          // std::cout << "child : " << (int)m_classifier["child"].at<float>(t,k) << std::endl;

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

#ifdef DEBUG
          std::cout << "ftrId = " << ftrId << ", ";
          std::cout << "ftrChnIdex = " << ftrChnIndex << ", ";
          std::cout << "zsA[ftrId] = " << zsA[ftrId] << ", ";
          std::cout << "ftrChnCol = " << ftrChnCol << ", ";
          std::cout << "csA[ftrId]+posWidth = " <<  csA[ftrId]+posWidth << ", ";
          std::cout << "ftrChnRow = " << ftrChnRow << ", ";
          std::cout << "rsA[ftrId]+posHeight = " <<  rsA[ftrId]+posHeight << ", ";
#endif

          // Obtain the feature value and threshold for the k-th tree node.
          ftr = channels[ftrChnIndex].at<float>(ftrChnRow, ftrChnCol);
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
        int min_ind[2];
        int max_ind[2];
        cv::minMaxIdx(costs_vector.rowRange(1,m_num_classes),
                      &min_positive_cost, &max_positive_cost, min_ind, max_ind, cv::Mat());
        // Assign the class of the min cost in costs_vector.
        h = min_ind[0] + 2; // +2 because the first positive class should have this label and the array is 0 index.

        // Obtain the cost for the negative class (the background).
        // The negative class is the first and the rest are the positives
        float neg_cost = costs_vector.at<float>(0,0);

        // Get the trace for the current detection window
        trace = -(min_positive_cost - neg_cost);

        if (trace <= m_clfData.cascThr) break;
      }

#ifdef DEBUG
      std::cout << "c = " << c << ", " << "r = " << r << ", " << "t = " << t << ", " << "h = " << h << std::endl;
#endif

      if (trace < 0)
      {
        h = 1;
      }

      if (h>1)
      {
        int index = c + (r * width1);
        cs[index] = c; 
        rs[index] = r; 
        hs1[index] = h;
        scores[index] = trace;
      }
    } // for (int k = rang.start; k < rang.end; k++)
  }); // parallel_for_ clossing.

  // Obtain detection rectangles
  std::vector<DetectionRectangle> detections;
  std::vector<float> scores_out;
  std::vector<float> labels;

  for( uint i=0; i < hs1.size(); i++ )
  {
    if (hs1[i] > 1) // hs1[i]>1 are object windows, hs1[i]==1 are background windows.
    {
      if (scores[i] > m_minScore)
      {
        DetectionRectangle det;
        det.bbox.x = cs[i] * m_clfData.stride;
        det.bbox.y = rs[i] * m_clfData.stride;
        det.bbox.width = m_clfData.modelDsPad.width;
        det.bbox.height = m_clfData.modelDsPad.height;
        det.score = scores[i];
        det.class_index = hs1[i];
        
        detections.push_back(det);
      }
    }
  }

  return detections;
}

void
BadacostDetector::showResults
  (
  cv::Mat img,
  const std::vector<DetectionRectangle>& detections
  )
{
  for(DetectionRectangle d: detections)
  {
    cv::rectangle(img, d.bbox, cv::Scalar(0, 255, 0), 2);

    // score with 2 decimal positions
    std::ostringstream out;
    out.precision(2);
    out << std::fixed << d.score;
    std::string score_txt = out.str();

    // The score is up left in the bbox rectangle
    cv::putText(img, //target image
                score_txt,
                cv::Point(d.bbox.x, d.bbox.y-5),
                cv::FONT_HERSHEY_DUPLEX,
                0.5,
                CV_RGB(255, 255, 0),
                1);

    // Class-1 shown in the left bottom corner
    cv::putText(img, //target image
                std::to_string(d.class_index-1),
                cv::Point(d.bbox.x, d.bbox.y+d.bbox.height+15),
                cv::FONT_HERSHEY_DUPLEX,
                0.5,
                CV_RGB(255, 255, 0),
                1);
  }
}





















