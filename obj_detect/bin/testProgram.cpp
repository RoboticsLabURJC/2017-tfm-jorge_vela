#include <iostream>
#include <detectors/BadacostDetector.h>
#include <pyramid/ChannelsPyramid.h>

#include <pyramid/ChannelsPyramidApproximatedStrategy.h>
#include <pyramid/ChannelsPyramidComputeAllStrategy.h>
#include <pyramid/ChannelsPyramidComputeAllParallelStrategy.h>
#include <pyramid/ChannelsPyramidApproximatedParallelStrategy.h>

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
        cout << "[Estrategia ChannelsPyramid] [Estrategia obtención características] [Fichero de salida de detecciones .yml] [Fichero salida estadísticas velocidad]" << endl;
        return 1;
    }

    std::string detect_strategy_str = argv[1];
    std::string acf_channels_impl_str = argv[2];

    if ((detect_strategy_str != "opencl") &&
        (detect_strategy_str != "all") &&
        (detect_strategy_str != "all_parallel") &&
        (detect_strategy_str != "approx") &&
        (detect_strategy_str != "approx_parallel"))
    {
        cout <<"ERROR EN LA FORMA DE INDICAR EL TIPO DE ESTRATEGIA. FORMATOS POSIBLES:" << endl;
        cout <<"opencl" << endl;
        cout <<"all" << endl;
        cout <<"all_parallel" << endl;
        cout <<"approx" << endl;
        cout <<"approx_parallel" << endl;
        return 1;
    }

    if ((acf_channels_impl_str != "pdollar") &&
        (acf_channels_impl_str != "opencv") &&
        (acf_channels_impl_str != "opencl"))
    {
        cout <<"ERROR EN LA FORMA DE INDICAR LA IMPLEMENTACIÓN DE CANALES ACF. FORMATOS POSIBLES:" << endl;
        cout <<"pdollar" << endl;
        cout <<"opencv" << endl;
        cout <<"opencl" << endl;
        return 1;
    }

    std::string clfPath = "obj_detect/tests/yaml/detectorComplete_2.yml";
    std::string filtersPath = "obj_detect/tests/yaml/filterTest.yml";

    cv::Mat image = cv::imread(argv[3], cv::IMREAD_COLOR);

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
#endif
    return 0;
}





