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


#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui.hpp>

using namespace std;

//#undef DEBUG
#define DEBUG

int main
  (
  int argc,
  char** argv
  )
{

    std::string clfPath = "obj_detect/tests/yaml/00_facesDetector_AFLW.yml";
    std::string filtersPath = "obj_detect/tests/yaml/00_filterTest_faces_AFLW.yml"; 
    BadacostDetector badacost("pdollar", "approx_parallel", 10);
    bool loadVal = badacost.load(clfPath, filtersPath); 
 
    // open the first webcam plugged in the computer
    cv::VideoCapture camera(0);
    if (!camera.isOpened()) {
        std::cerr << "ERROR: Could not open camera" << std::endl;
        return 1;
    }

    cv::namedWindow("Webcam", cv::WINDOW_AUTOSIZE);

    cv::Mat frame;
        
    camera >> frame;

    std::cout << frame.size() << std::endl;
    
    time_t start, end;
    time(&start);
    int nFrames = 0;
    while (1) {
        camera >> frame;
        cv::resize(frame,frame,cv::Size(640,360));

        std::vector<DetectionRectangle> detections = badacost.detect(frame);
        badacost.showResults(frame, detections);
        cv::imshow("Webcam", frame);
        if (cv::waitKey(1) >= 0)
            break;

        nFrames +=1;
        time(&end);
        double seconds = difftime (end, start);
        cout << "Time taken : " << nFrames/seconds << " fps" << endl;

    }
    return 0;
}














