#include <channels/ChannelsPyramid.h> 
#include <channels/Utils.h>
#include <channels/ChannelsExtractorLUV.h>
#include <channels/ChannelsExtractorGradMag.h>
#include <channels/ChannelsExtractorGradHist.h>

#include "gtest/gtest.h"
#include <opencv2/opencv.hpp>
#include <channels/Utils.h>

#include <iostream>

bool ChannelsPyramid::load(std::string opts){
//  bool loadValue = true;
  cv::FileStorage pPyramid;
  bool existOpts = pPyramid.open(opts, cv::FileStorage::READ);
  if(existOpts){
	  int nPerOct = pPyramid["nPerOct"]["data"][0];
      int nOctUp = 0; //pPyramid["nOctUp"]["data"][0];
	  int nApprox = pPyramid["nApprox"]["data"][0];
      //int pad[2] = {pPyramid["pad"]["data"][0], pPyramid["pad"]["data"][1]};
	  int shrink = pPyramid["pChns.shrink"]["data"];

	  m_nOctUp = nOctUp;  
	  m_nPerOct = nPerOct;
	  m_nApprox = nApprox;
	  m_shrink = shrink;

  }
  return existOpts;
}

std::vector<cv::Mat> ChannelsPyramid::getPyramid(cv::Mat img)
{
  Utils utils;
  int smooth = 1;
  ChannelsLUVExtractor channExtract{false, smooth};

  cv::Size sz = img.size();
  cv::Size minDs;
  minDs.width = 84; // <--- TODO: JM: Esto debería de venir del fichero del detector.
  minDs.height = 48; // <--- TODO: JM: Esto debería de venir de fichero del detector

  //int lambdas = {};

  //CONVERT I TO APPROPIATE COLOR SPACE-------------------------------------
  //img.convertTo(img, CV_32FC1);
  std::vector<cv::Mat> luvImage = channExtract.extractFeatures(img); //IMAGENES ESCALA DE GRISES??
  cv::Mat luv_image;
  cv::Mat luvImageChng;
  //printf("--> %f \n", luvImage[0].at<float>(152,140) );


  //luvImage[0].copyTo(luvImageChng);
  //luvImage[2].copyTo(luvImage[0]);
  //luvImageChng.copyTo(luvImage[2]);

  merge(luvImage, luv_image);


  //EN ESTAS LINEAS EL COMPRUEBA QUE SE CUMPLEN LOS REQUISITOS PARA LA CONVERSION
  cv::Mat imageUse = luv_image;

  

  //-------------------------------------------------------------------------

  //GET SCALES AT WHICH TO COMPUTE FEATURES---------------------------------

  //std::vector<float> scales;
  //scales = getScales(m_nPerOct, m_nOctUp, minDs, m_shrink, sz);

  std::vector<double> scales;
  std::vector<cv::Size2d> scaleshw;
  getScales(m_nPerOct, m_nOctUp, minDs, m_shrink, sz, scales, scaleshw);

  for (uint i=0; i < scales.size(); i++)
  {
    std::cout << scales[i] << ", ";
  }
  std::cout << std::endl;

  int nScales = scales.size();

  int isR;
  if(1){ //PREGUNTAR ESTE IF EN MATLAB
    isR = 1;
  }else{
    isR = 1+m_nOctUp*m_nPerOct;
  }
 
  std::vector<int> isRarr;
  int iLoop = 0;
  while(iLoop<nScales){
    isRarr.push_back(1 + iLoop);
    iLoop = iLoop + m_nApprox +1;
  }


  std::vector<int> isA;
  int valIsRArr = 0;
  for(int i = 0; i < nScales; i++){
    if(i+1 != isRarr[valIsRArr]){
      isA.push_back(i+1);
    }else{
      valIsRArr = valIsRArr + 1;
    }
  }

  std::vector<int> arrJ;
  arrJ.push_back(0); //[0] = 0;
  for(uint i = 0; i < isRarr.size() - 1; i++){
    arrJ.push_back((floor(isRarr[i] + isRarr[i+1]))/2);//[i+1] = (isRarr[i] + isRarr[i+1])/2;
  }
  arrJ.push_back(nScales); //[sizeisR+1]= nScales;

  std::vector<int> isN;
  for(uint i = 0; i <= isRarr.size(); i++){
    for(int j = arrJ[i]+1; j <= arrJ[i+1]; j++ ){
      //int val = i+j;
      isN.push_back(isRarr[i]);
    }
  }
  std::vector<cv::Mat> strucData[nScales];

  //COMPUTE IMAGE PYRAMID----------------------------------------------------
  std::vector<cv::Mat> pChnsCompute;
  for(int i=0; i< nScales/*28*//*isRarr.size()*/; i++){ //isRarr.size()
    float s=scales[i]; //[isRarr[i]-1];
    int sz_1 = round(sz.width*s/m_shrink)*m_shrink;
    int sz_2 = round(sz.height*s/m_shrink)*m_shrink;
    cv::Size sz1{sz_1, sz_2};
    //printf("ChnsPyramid 124 ; newSize; --->%d %d \n", sz1[0], sz1[1]);
    //printf("-->%d %d\n",sz1[0], sz1[1] );
    cv::Mat I1;
    if(sz.width == sz1.width && sz.height == sz1.height){
      I1 = imageUse;
    }else{
      I1 = utils.ImgResample(imageUse, sz1.width , sz1.height);
    }

    if(s==.5 && (m_nApprox>0 || m_nPerOct==1)){
      imageUse = I1;
    }
    std::string colorSpace = "LUV";


    pChnsCompute = utils.channelsCompute(I1, colorSpace.c_str(), m_shrink);
    strucData[i]/*[isRarr[i] - 1]*/ = pChnsCompute;
  } 
  cv::Mat data[pChnsCompute.size()][nScales];



  //std::vector<cv::Mat> data;
  //COMPUTE IMAGE PYRAMID [APPROXIMATE SCALES]-------------------------------
  /*for(int i=0; i< isA.size(); i++){
    int x = isA[i] -1;
    int iR =  isN[x];
    int sz_1 = round(sz[0]*scales[x]/m_shrink);
    int sz_2 = round(sz[1]*scales[x]/m_shrink);
    int sz1[2] = {sz_1, sz_2};
    for(int j=0; j < pChnsCompute.size(); j++){
      cv::Mat dataResample = pChnsCompute[j];
      std::vector<cv::Mat> resampleVect;
      for(int k = 0; k < strucData[iR-1].size(); k++){
        cv::Mat resample = utils.ImgResample(strucData[iR-1][k], sz1[0] , sz1[1]); //RATIO
        resampleVect.push_back(resample);
      }
      strucData[x] = resampleVect;
    }
  }*/
  

  //smooth channels, optionally pad and concatenate channels
  /*for(int i = 0; i < nScales; i++){
    for(int j=0; j < pChnsCompute.size();j++){
      data[j][i] = utils.convTri(luv_image, smooth);
    }
  }*/

  //FALTA EL OPTINALLY PAD
  //CONCATENA TODOS LOS CANALES
  std::vector<cv::Mat> channelsConcat;
  for(int i = 0; i < nScales; i++){
      cv::Mat concat;
      merge(strucData[i], concat);
      concat = utils.convTri(concat, 1);
      //cv::Mat dst;
      //cv::RNG rng(12345);
      //cv::Scalar value = cv::Scalar( rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255) );
      copyMakeBorder( concat, concat, 2, 2, 3, 3, cv::BORDER_REPLICATE, 0 );

      channelsConcat.push_back(concat);
  }
  return channelsConcat;
}

std::vector<cv::Mat> ChannelsPyramid::badacostFilters
  (
  cv::Mat pyramid,
  //std::string filterName
  std::vector<cv::Mat> filters
  )
{
  Utils utils;

/*
  //CARGAR EL FILTRO CREADO POR MATLAB DESDE UN YML
  cv::FileStorage filter;
  filter.open(filterName.c_str(), cv::FileStorage::READ);

  //OBTENER EL NOMBRE DE LOS DISTINTOS FILTROS PARA ESTE CASO
  std::vector<std::string> namesFilters;
  for(int i = 1; i < 5; i++){
    for(int j = 1; j< 11; j++){
      std::string name  = "filter_" + std::to_string(j) + "_" + std::to_string(i);
      namesFilters.push_back(name);
    }
  }
  //SE CARGAN LOS DISTINTOS FILTROS, CON LOS NOMBRES ANTERIORES DESDE EL YML
  std::vector<cv::Mat> filters;
  for(uint k = 0; k < namesFilters.size(); k++){
    cv::FileNode filterData = filter[namesFilters[k].c_str()]["data"];
    cv::FileNode filterRows = filter[namesFilters[k].c_str()]["rows"];
    cv::FileNode filterCols = filter[namesFilters[k].c_str()]["cols"];

    float* filt = new float[25*sizeof(float)];

    for(int i = 0; i < (int)filterRows; i++){
      for(int j = 0; j < (int)filterCols; j++){
        float x = (float)filterData[i*5+j];
        filt[i*5+j] = x;
      }
    }

    cv::Mat filterConver = cv::Mat(5,5, CV_32F, filt);
    transpose(filterConver,filterConver);

    //float *O = filterConver.ptr<float>();

    filters.push_back(filterConver);//(filt);
  }
 */
  int num_filters_per_channel = 4; // <-- TODO: JM: Estos números tienen que venir en el fichero yaml!
  int num_channels = 10; // <-- TODO: JM: Estos números tienen que venir en el fichero yaml!
  int filter_size = 5; // <-- TODO: JM: Estos números tienen que venir en el fichero yaml!

  //EJEMPLO PARA UNA ESCALA, QUE TIENE nChannels CANALES
  int nChannels = pyramid.channels();
  cv::Mat bgr_dst[nChannels];
  split(pyramid,bgr_dst);

  //SE REPITE UNA ESCALA PARA PASAR POR LOS FILTROS
  cv::Mat G;
  pyramid.copyTo(G);
  std::vector<cv::Mat> C_repMat;
  for(int i = 0; i < nChannels; i++){
    pyramid.copyTo(G);
    C_repMat.push_back(G);
  }

  //printf("pix 0,0 %f %f %d \n", (float)pyramid.at<float>(0,0), (float)pyramid.at<float>(0,1) , pyramid.size().height);
  //SE CONVOLUCIONA UNA IMAGEN CON LOS FILTROS Y SE OBTIENEN LAS IMAGENES DE SALIDA
  std::vector<cv::Mat> out_images;
  for(int j = 0; j < num_filters_per_channel; j++){
    cv::Mat splitted[nChannels];
    split(C_repMat[j],splitted);
    for(int i = 0; i < nChannels; i++){
      cv::Mat out_image; 
      //cv::Mat dst;
      //cv::RNG rng(12345);
      //cv::Scalar value = cv::Scalar( rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255) );
      //copyMakeBorder( splitted[i], dst, 2, 2, 3, 3, cv::BORDER_REFLECT, 0 );
      //printf("pix 0,0 %f %f %f %f %d \n", (float)dst.at<float>(2,3), (float)dst.at<float>(2,4), (float)dst.at<float>(2,2), (float)dst.at<float>(1,1), dst.size().height);

      filter2D( bgr_dst[i], out_image, CV_32FC1 , filters[i+(nChannels*j)], cv::Point( -1,-1 ), 0, cv::BORDER_CONSTANT );
      out_image = utils.ImgResample(out_image, round(out_image.size().width/2.0), round(out_image.size().height/2.0));

      //if(i == 1)
      //  printf("pix 0,0 %f %f %f \n", (float)out_image.at<float>(0,0), (float)out_image.at<float>(0,1) , (float)out_image.at<float>(1,0));

      out_images.push_back(out_image);
    }
  }
  //printf("ejem ejem.............\n");
  //cv::imshow("filtered", out_images[0]);
  //cv::waitKey(0);
  return out_images;
}


/**
 * Funcion getScales. En funcion de los parámetros de entrada retorna un vector con los distintos valores
 * por los que se tiene que escalar la imagen.
 *
 * @param nPerOct: Número de escalas por octava
 * @param nOctUp: Numero de octavas muestreadas para calcular
 * @param minDs: Tamaño mínimo de la imagen
 * @param shrink: Disminucion de la muestra para los canales
 * @param sz: Tamaño de la imagen
 *
 *
 */
int
ChannelsPyramid::getScales
  (
  int nPerOct,
  int nOctUp,
  const cv::Size& minDs,
  int shrink,
  const cv::Size& sz,
  std::vector<double>& scales,
  std::vector<cv::Size2d>& scaleshw
  )
{
  // set each scale s such that max(abs(round(sz*s/shrink)*shrink-sz*s)) is
  // minimized without changing the smaller dim of sz (tricky algebra)
  scales = {};
  scaleshw = {};

  if (!sz.area())
  {
      return 0;
  }

  cv::Size2d ratio(double(sz.width) / double(minDs.width), double(sz.height) / double(minDs.height));
  int nScales = std::floor(double(nPerOct) * (double(nOctUp) + log2(std::min(ratio.width, ratio.height))) + 1.0);

  double d0 = sz.height, d1 = sz.width;
  if (sz.height >= sz.width)
  {
      std::swap(d0, d1);
  }

  for (int i = 0; i < nScales; i++)
  {
    double s = std::pow(2.0, -double(i) / double(nPerOct) + double(nOctUp));
    double s0 = (std::round(d0 * s / shrink) * shrink - 0.25 * shrink) / d0;
    double s1 = (std::round(d0 * s / shrink) * shrink + 0.25 * shrink) / d0;
    std::pair<double, double> best(0, std::numeric_limits<double>::max());
    for (double j = 0.0; j < 1.0 - std::numeric_limits<double>::epsilon(); j += 0.01)
    {
      double ss = (j * (s1 - s0) + s0);
      double es0 = d0 * ss;
      es0 = std::abs(es0 - std::round(es0 / shrink) * shrink);
      double es1 = d1 * ss;
      es1 = std::abs(es1 - std::round(es1 / shrink) * shrink);
      double es = std::max(es0, es1);
      if (es < best.second)
      {
        best = { ss, es };
      }
    }
    scales.push_back(best.first);
  }

  auto tmp = scales;
  tmp.push_back(0);
  scales.clear();
  for (uint i = 1; i < tmp.size(); i++)
  {
    if (tmp[i] != tmp[i - 1])
    {
      double s = tmp[i - 1];
      scales.push_back(s);

      double x = std::round(double(sz.width) * s / shrink) * shrink / sz.width;
      double y = std::round(double(sz.height) * s / shrink) * shrink / sz.height;
      scaleshw.emplace_back(x, y);
    }
  }

  return 0;
}


/*
std::vector<float> ChannelsPyramid::getScales(  int nPerOct, int nOctUp, int minDs[], int shrink, int sz[]){

  float val1 = (float)sz[0]/(float)minDs[0];  
  float val2 = (float)sz[1]/(float)minDs[1];

  //printf("%f %f \n", val1, val2 );

  //float min = std::min(val1, val2);
  int nScales = floor(nPerOct*(nOctUp+log2(std::min(val1, val2)))+1);

  std::vector<float> scales;
  for(int i = 0; i< nScales; i++){
    float scalesValFloat = (-(float(i)/float(nPerOct))+float(nOctUp));
    float scalesVal = pow(2,scalesValFloat);
    scales.push_back(scalesVal);
  }

  //printf("%d\n", (int)scales.size());

  float d0 = 0;
  float d1 = 0;
  if(sz[0] < sz[1])
  {
    d0 = (float)sz[0];
    d1 = (float)sz[1];
  }
  else
  {
    d0 = (float)sz[1];
    d1 = (float)sz[0];
  }

  for (int i = 0; i <  nScales; i ++){
    float s = scales[i];

    float s0=(round(d0*s/ (float)shrink)* (float)shrink-.25* (float)shrink)/d0;
    float s1=(round(d0*s/ (float)shrink)* (float)shrink+.25* (float)shrink)/d0;

    float ssLoop = 0.0;

    std::vector<float> arrayPositions; //1 se sustituirá por abs(nScales)
   
    std::vector<float> ss;
    std::vector<float> es0;
    std::vector<float> es1;

    while(ssLoop < 1){
      float ssVal = ssLoop*(s1-s0)+s0;
      ss.push_back(ssVal);
      float es0val = d0*ssVal;  es0val=abs(es0val-round(es0val/shrink)*shrink);
      float es1val = d1*ssVal;  es1val=abs(es1val-round(es1val/shrink)*shrink);

      es0.push_back(es0val);
      es1.push_back(es1val);
      //printf("Valores --> %.4f %.4f %.4f \n", es0val, es1val, ssVal);
      ssLoop = ssLoop + 0.01;

    }

    std::vector<float> x = max(es0, es1);
    int pos = 0;
    float xMin = std::numeric_limits<float>::max(); 
    for(uint i=0; i < x.size(); i++){
      float valMax = ( es0[i] <  es1[i] ) ?  es1[i] : es0[i];
      //printf("%.4f %.4f %.4f \n",valMax, es0[i], es1[i]);
      if( valMax <= xMin){
        //printf("%.4f\n", x[i]);
        xMin = x[i];
        pos = i;
      }
    }
    scales[i] = ss[pos];
  }

  std::vector<float> kp;
  std::vector<float> scales2;
  for(uint i = 0; i < scales.size()-1; i++){
    //printf("--> %.4f %.4f\n",scales[i],  scales[i+1] );
    int kpVal = (scales[i] != scales[i+1]);
    if(kpVal == 1){
      scales2.push_back(scales[i]);
    }
    kp.push_back(kpVal);
  }
  scales2.push_back(scales[scales.size()-1]);


  //std::vector<float> scaleshw;
  int sizeScaleshw = scales2.size();
  float scaleshw[sizeScaleshw][sizeScaleshw];
  for(int i = 0; i < sizeScaleshw; i++){
    float h = round(sz[0]*scales2[i]/shrink)*shrink/sz[0];
    float w = round(sz[1]*scales2[i]/shrink)*shrink/sz[1];
    scaleshw[i][0] = h;
    scaleshw[i][1] = w;
  }

  return scales2;
}
*/









