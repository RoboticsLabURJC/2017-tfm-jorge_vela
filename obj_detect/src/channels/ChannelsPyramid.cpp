

#include <channels/ChannelsPyramid.h> 
#include <channels/Utils.h>
#include <channels/ChannelsExtractorLUV.h>
#include <channels/ChannelsExtractorGradMag.h>
#include <channels/ChannelsExtractorGradHist.h>

#include "gtest/gtest.h"
#include <opencv/cv.hpp>
#include <channels/Utils.h>

#include <iostream>

bool ChannelsPyramid::load(std::string opts){
  bool loadValue = true;
  cv::FileStorage pPyramid;
  bool existOpts = pPyramid.open(opts, cv::FileStorage::READ);
  if(existOpts){
	  int nPerOct = pPyramid["nPerOct"]["data"][0];
	  int nOctUp = pPyramid["nOctUp"]["data"][0];
	  int nApprox = pPyramid["nApprox"]["data"][0];
	  int pad[2] = {pPyramid["pad"]["data"][0], pPyramid["pad"]["data"][1]};
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

  luvImage[0].copyTo(luvImageChng);
  luvImage[2].copyTo(luvImage[0]);
  luvImageChng.copyTo(luvImage[2]);



  merge(luvImage, luv_image);

  //EN ESTAS LINEAS EL COMPRUEBA QUE SE CUMPLEN LOS REQUISITOS PARA LA CONVERSION
  cv::Mat imageUse = luv_image;
 

  //-------------------------------------------------------------------------

  //GET SCALES AT WHICH TO COMPUTE FEATURES---------------------------------
  std::vector<float> scales;
 
  scales = getScales(m_nPerOct, m_nOctUp, minDs, m_shrink, sz);

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
  for(int i = 0; i < isRarr.size() - 1; i++){
    arrJ.push_back((floor(isRarr[i] + isRarr[i+1]))/2);//[i+1] = (isRarr[i] + isRarr[i+1])/2;
  }
  arrJ.push_back(nScales); //[sizeisR+1]= nScales;

  std::vector<int> isN;
  for(int i = 0; i <= isRarr.size(); i++){
    for(int j = arrJ[i]+1; j <= arrJ[i+1]; j++ ){
      int val = i+j;
      isN.push_back(isRarr[i]);
    }
  }
  std::vector<cv::Mat> strucData[nScales];
  //COMPUTE IMAGE PYRAMID----------------------------------------------------
  std::vector<cv::Mat> pChnsCompute;
  for(int i=0; i< isRarr.size(); i++){
    float s=scales[isRarr[i]-1];
    int sz_1 = round(sz[0]*s/m_shrink)*m_shrink;
    int sz_2 = round(sz[1]*s/m_shrink)*m_shrink;
    int sz1[2] = {sz_1, sz_2};
    //printf("-->%d %d\n",sz1[0], sz1[1] );
    cv::Mat I1;
    if(sz[0] == sz1[0] && sz[1] == sz1[1]){
      I1 = imageUse;
    }else{
      I1 = utils.ImgResample(imageUse, sz1[0] , sz1[1] );
    }

    if(s==.5 && (m_nApprox>0 || m_nPerOct==1)){
      imageUse = I1;
    }
    pChnsCompute = utils.channelsCompute(I1,m_shrink);
    strucData[isRarr[i] - 1] = pChnsCompute;
  } 

  cv::Mat data[pChnsCompute.size()][nScales];


  //SUPONEMOS QUE NO SE DARÁ ESTE CASO---------------------------------------
  // if lambdas not specified compute image specific lambdas
  if( nScales>0 && m_nApprox>0){
    std::vector<int> isA;
    for(int i = m_nOctUp*m_nPerOct; i < nScales; i){
      isA.push_back(i);
      i = i+m_nApprox+1;
    }

    int nTypes = 3;  // Se utiliza por cada canal que se añade en chnsCompute, para nosotros 10
    
    /*std::vector<int> f0;
    for(int i = 0; i < nTypes; i++){
      f0.push_back(0);
    }

    std::vector<int> f1 = f0;*/


    double f0[] = {0,0,0};
    double f1[] = {0,0,0};
    double lambdas[] = {0,0,0};
    for(int i = 0; i < nTypes; i++){
      int w = strucData[isA[0]][i].size().width;
      int h = strucData[isA[0]][i].size().height;

      double sum1; 
      double sum2;
      double sum3;
      for(int j = 0; j < 3; j++){
        double a1 = cv::sum(strucData[isA[0]][i])[0];
        f0[0]+=a1;   
        double a2 = cv::sum(strucData[isA[1]][i])[0];
        f1[0]+=a2;    
      }

      f0[0] = f0[0]/(w*h*3);
      f1[0] = f1[0]/(w*h*3);

      f0[1] =  cv::sum(strucData[isA[0]][3])[0]/(w*h);
      f1[1] =  cv::sum(strucData[isA[1]][3])[0]/(w*h);

      for(int j = 4; j < strucData[isA[0]].size(); j++){
        double a1 = cv::sum(strucData[isA[0]][i])[0];
        f0[2]+=a1;  
        double a2 = cv::sum(strucData[isA[0]][i])[0];
        f1[2]+=a2;
      }

      f0[2]=f0[2]/(w*h*(strucData[isA[0]].size() - 4));
      f1[2]=f1[2]/(w*h*(strucData[isA[0]].size() - 4)); 

    }

    for(int i=0; i< 3; i++){
      printf("%d %d \n", scales[isA[0]], scales[isA[1]] );
      printf("%f %f \n", f0[i], f1[i] );


      lambdas[i] = -log2 (f0[i]/f1[i]) / log2(scales[isA[0]]/scales[isA[1]]);
    }
    printf("%f %f %f \n", lambdas[0], lambdas[1], lambdas[2]); //REVISAR LOS RESULTADOS,,,??????????????????????????????????????????????????
    
  }


  //std::vector<cv::Mat> data;
  //COMPUTE IMAGE PYRAMID [APPROXIMATE SCALES]-------------------------------
  
  for(int i=0; i< isA.size(); i++){
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
  }
  

  //smooth channels, optionally pad and concatenate channels
  /*for(int i = 0; i < nScales; i++){
    for(int j=0; j < pChnsCompute.size();j++){
      data[j][i] = convTri(luv_image, 1);
    }
  }*/
  //FALTA EL OPTINALLY PAD
  //CONCATENA TODOS LOS CANALES
  std::vector<cv::Mat> channelsConcat;
  for(int i = 0; i < nScales; i++){
      cv::Mat concat;
      merge(strucData[i], concat);
      concat = utils.convTri(concat, 1);
      channelsConcat.push_back(concat);
  }

  return channelsConcat;
}



std::vector<cv::Mat> ChannelsPyramid::badacostFilters(std::vector<cv::Mat> pyramid, std::string filterName){ //ChannelsPyramid::
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
  for(int k = 0; k < namesFilters.size(); k++){
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
    float *O = filterConver.ptr<float>();

    filters.push_back(filterConver);//(filt);
  }

  //EJEMPLO PARA UNA ESCALA, QUE TIENE nChannels CANALES
  int nChannels = pyramid[0].channels();
  cv::Mat bgr_dst[nChannels];
  split(pyramid[0],bgr_dst);

  //SE REPITE UNA ESCALA PARA PASAR POR LOS FILTROS
  cv::Mat G;
  pyramid[0].copyTo(G);
  std::vector<cv::Mat> C_repMat;
  for(int i = 0; i < nChannels; i++){
    pyramid[0].copyTo(G);
    C_repMat.push_back(G);
  }

  //SE CONVOLUCIONA UNA IMAGEN CON LOS FILTROS Y SE OBTIENEN LAS IMAGENES DE SALIDA
  std::vector<cv::Mat> out_images;
  for(int j = 0; j < 4; j++){
    cv::Mat splitted[nChannels];
    split(C_repMat[j],splitted);
    for(int i = 0; i < nChannels; i++){
      cv::Mat out_image; 
      filter2D(splitted[i], out_image, -1 , filters[i+(nChannels*j)], cv::Point( -1, -1 ), 0, cv::BORDER_REFLECT );
      out_images.push_back(out_image);
    }
  }
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
  if(sz[0]==0 || sz[1]==0)
  {
    int scales[0];
    int scaleshw[0];
  }
  
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
    for(int i=0; i < x.size(); i++){
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
  for(int i = 0; i < scales.size()-1; i++){
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










