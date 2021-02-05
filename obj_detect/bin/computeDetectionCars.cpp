#include <iostream>
#include <detectors/BadacostDetector.h>
#include <pyramid/ChannelsPyramid.h>

#include <pyramid/ChannelsPyramidApproximatedStrategy.h>
#include <pyramid/ChannelsPyramidComputeAllStrategy.h>
#include <pyramid/ChannelsPyramidComputeAllParallelStrategy.h>
#include <pyramid/ChannelsPyramidApproximatedParallelStrategy.h>

#include <bits/stdc++.h> 
#include <iostream> 
#include <sys/stat.h> 
#include <sys/types.h> 

using namespace std;

//#undef DEBUG
#define DEBUG

int main
  (
  int argc,
  char** argv
  )
{
    if (argc != 5)
    {
        cout << "Error en la introducción de argumentos." << endl;
        //cout << "[Estrategia ChannelsPyramid] [Imagen entrada] [Fichero de salida de detecciones .yml] [Fichero salida estadísticas velocidad]" << endl;
        cout << "[Estrategia ChannelsPyramid] [Estrategia obtención características] [Carpeta de entrada] [Carpeta de salida]" << endl;
        return 1;
    }

    std::string detect_strategy_str = argv[1];
    std::string acf_channels_impl_str = argv[2];

    if ((detect_strategy_str != "all") &&
            (detect_strategy_str != "all_parallel") &&
            (detect_strategy_str != "approx") &&
            (detect_strategy_str != "approx_parallel"))
    {
        cout <<"ERROR EN LA FORMA DE INDICAR EL TIPO DE ESTRATEGIA. FORMATOS POSIBLES:" << endl;
        cout <<"all" << endl;
        cout <<"all_parallel" << endl;
        cout <<"approx" << endl;
        cout <<"approx_parallel" << endl;
        return 1;
    }

    if ((acf_channels_impl_str != "pdollar") &&
            (acf_channels_impl_str != "opencv"))
    {
        cout <<"ERROR EN LA FORMA DE INDICAR LA IMPLEMENTACIÓN DE CANALES ACF. FORMATOS POSIBLES:" << endl;
        cout <<"pdollar" << endl;
        cout <<"opencv" << endl;
        return 1;
    }

    std::string clfPath = "obj_detect/tests/yaml/detectorComplete_2.yml";
    std::string filtersPath = "obj_detect/tests/yaml/filterTest.yml";

    BadacostDetector badacost(detect_strategy_str, acf_channels_impl_str);
    bool loadVal = badacost.load(clfPath, filtersPath); //, pyrPath (segundo parametro)
    float angle[20] = {-0.0785 ,-0.3142 ,-0.6283 ,-0.9425 ,-1.2566 ,-1.5708 ,-1.8850 ,-2.1991 ,-2.5133 ,-2.8274, -3.1416, 2.8274, 2.5133, 2.1991, 1.8850, 1.5708, 1.2566, 0.9425, 0.6283, 0.3142};

    std::string folder = argv[3];
    std::string folderOut = argv[4];

    if (mkdir(folderOut.c_str(), 0777) == -1) 
        cerr << "Error :  " << strerror(errno) << endl;

    for(int i = 6733; i < 7481; i++){
      std::string nameImg = folder + "/00" + to_string(i) + ".png";
      printf("%s\n",nameImg.c_str() );
      cv::Mat image = cv::imread(nameImg, cv::IMREAD_COLOR);
      std::vector<DetectionRectangle> detections = badacost.detect(image);
      std::cout << detections << std::endl;
      ofstream myfile;
      std::string nameFile = folderOut + "/00" + to_string(i) + ".txt";
      myfile.open(nameFile);
      for(int i = 0; i < detections.size(); i++){
        float ang = angle[detections[i].class_index - 2];
        myfile << "Car -1 -1 " + to_string(ang) + " "  + to_string(detections[i].bbox.x) + " " + to_string(detections[i].bbox.y) + " " + to_string(detections[i].bbox.x + detections[i].bbox.width)
        + " "  + to_string(detections[i].bbox.y + detections[i].bbox.height) + " -1 -1 -1 -1000 -1000 -1000 -10 "+ to_string(detections[i].score) << endl;
      }
      myfile.close();
    }


/*  cv::Mat image = cv::imread(argv[3], cv::IMREAD_COLOR);

    BadacostDetector badacost(detect_strategy_str, acf_channels_impl_str);
    bool loadVal = badacost.load(clfPath, filtersPath); //, pyrPath (segundo parametro)
    if (!loadVal)
    {
        cout <<"FICHERO CON EL CLASIFICADOR NO ENCONTRADO." << endl;
    }

    auto start = std::chrono::system_clock::now(); 
    std::vector<DetectionRectangle> detections = badacost.detect(image);
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<float,std::milli> duration = end - start;
    std::cout << duration.count() << " time in detections" << std::endl;

    cv::FileStorage fs(argv[4], cv::FileStorage::WRITE);
    for(uint i = 0; i < detections.size(); i++)
    {
        std::string a = "Detections_" + to_string(i);
        fs << a << "{";
        fs << "bbox" << detections[i].bbox;
        fs << "score" << detections[i].score;
        fs << "class_index" << detections[i].class_index;

        fs << "}";
    }
    fs.release();
#ifdef DEBUG
    std::cout << detections;
    badacost.showResults(image, detections);
    cv::imshow("image", image);
    cv::waitKey();
#endif*/
    return 0;
}





