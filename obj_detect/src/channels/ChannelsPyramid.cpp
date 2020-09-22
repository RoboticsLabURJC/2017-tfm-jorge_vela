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
//	  int pad[2] = {pPyramid["pad"]["data"][0], pPyramid["pad"]["data"][1]};
	  int shrink = pPyramid["pChns.shrink"]["data"];

	  m_nOctUp = nOctUp;  
	  m_nPerOct = nPerOct;
	  m_nApprox = nApprox;
	  m_shrink = shrink;

  }
  return existOpts;
}

std::vector<cv::Mat> ChannelsPyramid::getPyramid(cv::Mat img){ 
  Utils utils;
  int smooth = 1;
  ChannelsLUVExtractor channExtract{false, smooth};


  int sz[2] = {img.size().width, img.size().height}; //SE HA CAMBIADO ESTO PORQUE HABIA UN ERROR EN EL ORDEN
  //int shrink = 4;
  int minDs[2] = {84,48}; //{minDsA[0], minDsA[1]}; //DEBIDO AL CAMBIO DE LA ANTERIOR LINEA SE MODIFICA ESTA

  //int lambdas = {};

  //CONVERT I TO APPROPIATE COLOR SPACE-------------------------------------
  std::vector<cv::Mat> luvImage = channExtract.extractFeatures(img); //IMAGENES ESCALA DE GRISES??
  cv::Mat luv_image;
  cv::Mat luvImageChng;

  //luvImage[0].copyTo(luvImageChng);
  //luvImage[2].copyTo(luvImage[0]);
  //luvImageChng.copyTo(luvImage[2]);


  merge(luvImage, luv_image);

  //EN ESTAS LINEAS EL COMPRUEBA QUE SE CUMPLEN LOS REQUISITOS PARA LA CONVERSION
  cv::Mat imageUse = luv_image;

  //-------------------------------------------------------------------------

  //GET SCALES AT WHICH TO COMPUTE FEATURES---------------------------------
  std::vector<float> scales;
 
  scales = getScales(m_nPerOct, m_nOctUp, minDs, m_shrink, sz);
  //printf("ChnsPyramid 73 shrink --> %d \n", m_shrink );

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
    int sz_1 = round(sz[0]*s/m_shrink)*m_shrink;
    int sz_2 = round(sz[1]*s/m_shrink)*m_shrink;
    int sz1[2] = {sz_1, sz_2};
    //printf("ChnsPyramid 124 ; newSize; --->%d %d \n", sz1[0], sz1[1]);
    //printf("-->%d %d\n",sz1[0], sz1[1] );
    cv::Mat I1;
    if(sz[0] == sz1[0] && sz[1] == sz1[1]){
      I1 = imageUse;
    }else{
      I1 = utils.ImgResample(imageUse, sz1[0] , sz1[1]);
    }

    if(s==.5 && (m_nApprox>0 || m_nPerOct==1)){
      imageUse = I1;
    }
    std::string colorSpace = "LUV";

    //cv::imshow("", I1);
    //cv::waitKey(0);

    pChnsCompute = utils.channelsCompute(I1, colorSpace.c_str(), m_shrink);
    strucData[i]/*[isRarr[i] - 1]*/ = pChnsCompute;
  } 
  cv::Mat data[pChnsCompute.size()][nScales];

  /*cv::imshow("1", strucData[7][0]);
  cv::imshow("2", strucData[7][1]);
  cv::imshow("3", strucData[7][2]);
  cv::imshow("4", strucData[7][3]);
  cv::imshow("5", strucData[7][4]);
  cv::imshow("6", strucData[7][5]);
  cv::imshow("7", strucData[7][6]);
  cv::imshow("8", strucData[7][7]);
  cv::imshow("9", strucData[7][8]);
  cv::imshow("10", strucData[7][9]);

  cv::waitKey(0);*/


  /*
  //SUPONEMOS QUE NO SE DARÁ ESTE CASO---------------------------------------
  // if lambdas not specified compute image specific lambdas
  double lambdas[] = {0,0,0};
  if( nScales>0 && m_nApprox>0){
    std::vector<int> isA;
    for(int i = m_nOctUp*m_nPerOct; i < nScales; i){
      isA.push_back(i);
      i = i+m_nApprox+1;
    }

    int nTypes = 3;  // Se utiliza por cada canal que se añade en chnsCompute, para nosotros 10

    double f0[] = {0,0,0};
    double f1[] = {0,0,0};
    
    //for(int i = 0; i < nTypes; i++){
    int w = strucData[isA[0]][0].size().width;
    int h = strucData[isA[0]][0].size().height;

    int w1 = strucData[isA[1]][3].size().width;
    int h1 = strucData[isA[1]][3].size().height;

    double sum1; 
    double sum2;
    double sum3;
    for(int j = 0; j < 3; j++){
      double a1 = cv::sum(strucData[isA[0]][j])[0]/(w*h*3);
      f0[0]+=a1;   
      double a2 = cv::sum(strucData[isA[1]][j])[0]/(w1*h1*3);
      f1[0]+=a2;    
    }

    f0[1] =  cv::sum(strucData[isA[0]][3])[0]/(w*h);
    f1[1] =  cv::sum(strucData[isA[1]][3])[0]/(w1*h1);

    for(int j = 4; j < strucData[isA[0]].size(); j++){
      double a1 = cv::sum(strucData[isA[0]][j])[0];
      f0[2]+=a1;  
      double a2 = cv::sum(strucData[isA[1]][j])[0];
      f1[2]+=a2;
    }

    f0[2]=f0[2]/(w*h*(strucData[isA[0]].size() - 4));
    f1[2]=f1[2]/(w1*h1*(strucData[isA[1]].size() - 4)); 

    //}
    for(int i=0; i< 3; i++){
      lambdas[i] = -log2 (f0[i]/f1[i]) / log2(scales[isA[0]]/scales[isA[1]]);
    }
    printf("%f %f %f \n", lambdas[0], lambdas[1], lambdas[2]); //REVISAR LOS RESULTADOS,,,??????????????????????????????????????????????????
    
  }*/


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
      //copyMakeBorder( concat, concat, 2, 2, 3, 3, cv::BORDER_REPLICATE, 0 );


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
  for(int j = 0; j < 4; j++){
    cv::Mat splitted[nChannels];
    split(C_repMat[j],splitted);
    for(int i = 0; i < nChannels; i++){
      cv::Mat out_image; 
      //filter2D(splitted[i], out_image, CV_32FC1 , filters[i+(nChannels*j)], cv::Point( 0,0 ), 0, cv::BORDER_REFLECT );

      cv::Mat dst;
      cv::RNG rng(12345);
      //cv::Scalar value = cv::Scalar( rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255) );
      copyMakeBorder( splitted[i], dst, 2, 2, 3, 3, cv::BORDER_REFLECT, 0 );
      //printf("pix 0,0 %f %f %f %f %d \n", (float)dst.at<float>(2,3), (float)dst.at<float>(2,4), (float)dst.at<float>(2,2), (float)dst.at<float>(1,1), dst.size().height);

      filter2D( dst, out_image, CV_32FC1 , filters[i+(nChannels*j)], cv::Point( 0,0 ), 0, cv::BORDER_CONSTANT );
      out_image = utils.ImgResample(out_image, round((float)out_image.size().width/2), round((float)out_image.size().height/2));

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
std::vector<float> ChannelsPyramid::getScales(  int nPerOct, int nOctUp, int minDs[], int shrink, int sz[]){
  /*if(sz[0]==0 || sz[1]==0)
  {
    int scales[0];
    int scaleshw[0];
  }
  */
  
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

  /*for(int i=0; i< scales.size(); i++){
    printf("scale: %.4f\n", scales[i] );
  }*/

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
  /*for(int i =0; i < sizeScaleshw; i++){
    printf("%.4f \n", scales2[i]);// scaleshw[i][0], scaleshw[i][1]);
    //printf("%.4f %.4f \n", scaleshw[i][0], scaleshw[i][1]);
  }*/

  return scales2;
}










