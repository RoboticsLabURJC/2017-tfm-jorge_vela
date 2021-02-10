#include <iostream>
#include <detectors/BadacostDetector.h>
#include <pyramid/ChannelsPyramid.h>

#include <pyramid/ChannelsPyramidApproximatedStrategy.h>
#include <pyramid/ChannelsPyramidComputeAllStrategy.h>
#include <pyramid/ChannelsPyramidComputeAllParallelStrategy.h>
#include <pyramid/ChannelsPyramidApproximatedParallelStrategy.h>

#include <chrono>
using namespace std;

#include <iostream>
#include <dirent.h>
#include <sys/types.h>

#undef DEBUG

std::string getFileExt(const string& s) {

   size_t i = s.rfind('.', s.length());
   if (i != string::npos) {
      return(s.substr(i+1, s.length() - i));
   }

   return("");
}

std::string getName(std::string nameToSplit){

  std::istringstream iss(nameToSplit);
  std::string segment;
  std::vector<std::string> seglist;
  while(std::getline(iss, segment, '/'))
  {
     seglist.push_back(segment); //Spit string at '_' character
  }

  std::istringstream iss2(seglist[6]);
  std::string segment2;
  std::vector<std::string> seglist2;
  while(std::getline(iss2, segment2, '_'))
  {
     seglist2.push_back(segment2); //Spit string at '_' character
  }

  std::istringstream iss3(seglist2[1]);
  std::string segment3;
  std::vector<std::string> seglist3;
  while(std::getline(iss3, segment3, '.'))
  {
     seglist3.push_back(segment3); //Spit string at '_' character
  }

  std::string tot = seglist[2] + "_00" + seglist3[0]; 

  return tot;

}


int main(int argc, char** argv) {

    if (argc != 5 && argc != 6)
    {
        cout << "Error en la introducción de argumentos." << endl;
        //cout << "[Estrategia ChannelsPyramid] [Imagen entrada] [Fichero de salida de detecciones .yml] [Fichero salida estadísticas velocidad]" << endl;
        cout << "[Estrategia ChannelsPyramid] [Estrategia obtención características] [Fichero de salida de detecciones] [Base de datos utilizada] [(fddb fold)]" << endl;
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

	std::string clfPath = "obj_detect/tests/yaml/00_facesDetector_AFLW.yml";
	std::string filtersPath = "obj_detect/tests/yaml/00_filterTest_faces_AFLW.yml"; 
    BadacostDetector badacost(detect_strategy_str, acf_channels_impl_str, 10);
    bool loadVal = badacost.load(clfPath, filtersPath); 

	ofstream myfile;
    myfile.open (argv[3]);

    std::string typeDataBase = argv[4];
    if(typeDataBase == "afw"){
	    struct dirent *entry;
	    DIR *dir = opendir("afw");
	    if (dir == NULL) {
	        printf("ERROR CARPETA\n");
	       	return 1;
	    }

	    while ((entry = readdir(dir)) != NULL) {
	    	std::string ext =  getFileExt(entry->d_name);
	    	if(ext == "jpg"){
	    		std::string name = entry->d_name;
	    		std::cout << name << std::endl;
	    		std::string nameTot = "afw/" + name;
	    		cv::Mat image = cv::imread(nameTot, cv::IMREAD_COLOR);
	    		std::vector<DetectionRectangle> detections = badacost.detect(image);
	    		//std::cout << detections << std::endl;

	    		for(int i = 0; i < detections.size(); i++){
	    			myfile << name + "," + to_string(detections[i].bbox.x) + "," + to_string(detections[i].bbox.y) + "," + 
	    			to_string(detections[i].bbox.x + detections[i].bbox.width) + ","  + to_string(detections[i].bbox.y + detections[i].bbox.height) + "," + to_string(detections[i].score) + ",-"<< std::endl;
					
					std::string token = name.substr(0, name.find("."));
	    			std::string nameFile = "afw_results/" + token + "_" +to_string(i) + ".pts";
	    			ofstream myfileSpec;
	    			myfileSpec.open(nameFile);

	    			myfileSpec << to_string(detections[i].bbox.x) + "," + to_string(detections[i].bbox.y) + "," + 
	    			to_string(detections[i].bbox.x + detections[i].bbox.width) + ","  + to_string(detections[i].bbox.y + detections[i].bbox.height) + to_string(detections[i].score) + ",-"<< std::endl;

	    			myfileSpec.close();
	    		}
		    	//left_x top_y width height detection_score
#ifdef DEBUG
			    badacost.showResults(image, detections);
			    cv::imshow("image", image);
			    cv::waitKey();
#endif
	    	}
	    }
	    closedir(dir);
  	}

    if(typeDataBase == "fddb"){
    	if(argc != 6){
    	  std::cout << "Falta nombre de los ficheros FDDB" <<std::endl;
    	  return 1;
    	}

    	std::string nameFiles = argv[5];
    	std::ifstream file(nameFiles);
		std::string str; 
		while (std::getline(file, str)) {
		  std::string nameTot = "fddb/originalPics/"+ str + ".jpg";
		  std::cout << nameTot << "\n";
		  cv::Mat image = cv::imread(nameTot, cv::IMREAD_COLOR);

		  std::cout << image.size() << std::endl;

		  std::vector<DetectionRectangle> detections = badacost.detect(image);

		  

		  std::string namePic = getName(nameTot);
		  //myfile << str << std::endl;
		  //myfile << detections.size() << std::endl;
		  for(int i = 0; i < detections.size(); i++){
		  	myfile << namePic + " " + to_string(detections[i].score) + " " + to_string(detections[i].bbox.x) + " " + to_string(detections[i].bbox.y) + " " + to_string(detections[i].bbox.x + detections[i].bbox.width) 
			+ " " + to_string(detections[i].bbox.y + detections[i].bbox.height) << std::endl;
			//myfile << to_string(detections[i].bbox.x) + " " + to_string(detections[i].bbox.y) + " " + to_string(detections[i].bbox.x + detections[i].bbox.width) 
			//+ " " + to_string(detections[i].bbox.y + detections[i].bbox.height) + " " + to_string(detections[i].score) << std::endl;
		  }

#ifdef DEBUG
			    badacost.showResults(image, detections);
			    cv::imshow("image", image);
			    cv::waitKey();
#endif
		}
    }

    return 0;
}





