#include <iostream>
#include <detectors/BadacostDetector.h>
#include <pyramid/ChannelsPyramid.h>

#include <pyramid/ChannelsPyramidApproximatedStrategy.h>
#include <pyramid/ChannelsPyramidComputeAllStrategy.h>
#include <pyramid/ChannelsPyramidComputeAllParallelStrategy.h>
#include <pyramid/ChannelsPyramidApproximatedParallelStrategy.h>

using namespace std;

int main(int argc, char** argv) {

	if(argc != 5){
		cout << "Error en la introducción de argumentos." << endl;
		cout << "[Estrategia ChannelsPyramid] [Imagen entrada] [Fichero de salida de detecciones .yml] [Fichero salida estadísticas velocidad]" << endl;
		return 1;
	}


	ChannelsPyramid* pPyramidStrategy;
	if(std::string(argv[1]) == "allparallel"){
		pPyramidStrategy = dynamic_cast<ChannelsPyramid*>( new ChannelsPyramidComputeAllParallelStrategy() );
	}else if(std::string(argv[1]) == "aprox"){
		pPyramidStrategy = dynamic_cast<ChannelsPyramid*>( new ChannelsPyramidApproximatedStrategy() );
	}else if(std::string(argv[1]) == "aproxparallel"){
		pPyramidStrategy = dynamic_cast<ChannelsPyramid*>( new ChannelsPyramidApproximatedParallelStrategy() );
	}else if(std::string(argv[1]) == "all"){
		pPyramidStrategy = dynamic_cast<ChannelsPyramid*>( new ChannelsPyramidComputeAllStrategy() );
	}else{
		cout <<"ERROR EN LA FORMA DE INDICAR EL TIPO DE ESTRATEGIA. FORMATOS POSIBLES:" << endl;
		cout <<"all" << endl;
		cout <<"allparallel" << endl;
		cout <<"aprox" << endl;
		cout <<"aproxparallel" << endl;
		return 1;
	}


	std::string clfPath = "obj_detect/tests/yaml/detectorComplete_2.yml";
	std::string filtersPath = "obj_detect/tests/yaml/filterTest.yml";

	cv::Mat image = cv::imread(argv[2], cv::IMREAD_COLOR);

	BadacostDetector badacost(pPyramidStrategy);
	bool loadVal = badacost.load(clfPath, filtersPath); //, pyrPath (segundo parametro)
	std::vector<DetectionRectangle> detections = badacost.detect(image);

	cv::FileStorage fs(argv[3], cv::FileStorage::WRITE);
	for(int i = 0; i < detections.size(); i++){
		std::string a = "Detections_" + to_string(i);
	    fs << a << "{";
	    fs << "bbox" << detections[i].bbox;
	    fs << "score" << detections[i].score;
	    fs << "class_index" << detections[i].class_index;

	    fs << "}"; 
	}
	fs.release();  


    return 0;

}





