#include <iostream>
#include <string>
#include <detectors/BadacostDetector.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>

int main
  (
  int argc,
  char** argv
  )
{
  const char* keys =
          "{ h help      |     | print help message }"
          "{ c camera    | 0   | capture video from camera (device index starting from 0) }"
          "{ f features  | pdollar  | features (supported: pdollar, opencv, opencl)}"
          "{ p pyramid   | approximated_parallel  | pyramid (supported: all, all_parallel, approximated, approximated_parallel, packed_img)}"
          "{ d detector  |  ../tests/yaml/00_facesDetector_AFLW.yml   | detector file}"
          "{ t filters   |  ../tests/yaml/00_filterTest_faces_AFLW.yml | filters file}"
          "{ m cpu       |     | run without OpenCL }"
          "{ v video     |     | use video as input }"
          "{ o original  |     | use original frame size (do not resize to 320x240)}"
          ;

  cv::CommandLineParser parser(argc, argv, keys);
  parser.about("This sample demonstrates using BAdaCost face detection.");
  if (parser.has("help"))
  {
      parser.printMessage();
      return 0;
  }
  int camera = parser.get<int>("camera");
  std::string features_impl = parser.get<std::string>("features");
  std::string pyramid_strategy = parser.get<std::string>("pyramid");
  std::string detector_file = parser.get<std::string>("detector");
  std::string filters_file = parser.get<std::string>("filters");
  bool useCPU = parser.has("cpu");
  std::string filename = parser.get<std::string>("video");
  bool useOriginalSize = parser.has("original");
  if (!parser.check())
  {
    parser.printErrors();
    return 1;
  }

  if (!useCPU)
  {
    cv::ocl::Context context;
    if (!context.create(cv::ocl::Device::TYPE_GPU))
    {
      std::cout << "Failed creating OpenCL context..." << std::endl;
    }
    cv::ocl::Device(context.device(0));
    cv::ocl::setUseOpenCL(true);
  }

  BadacostDetector badacost(features_impl, pyramid_strategy, 10);
  bool loadVal = badacost.load(detector_file, filters_file);
 
  // open the first webcam plugged in the computer
  cv::VideoCapture cap;
  if (filename.empty())
  {
      cap.open(camera);
  }
  else
  {
      cap.open(filename);
  }

  if (!cap.isOpened())
  {
      std::cout << "Can not open video stream: '" << (filename.empty() ? "<camera>" : filename) << "'" << std::endl;
      return 2;
  }

  // Set properties. Each returns === True on success (i.e. correct resolution)
  if (!useOriginalSize)
  {
//    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
//    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 320);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 240);
  }


  cv::namedWindow("Webcam", cv::WINDOW_AUTOSIZE);

  // Print frame size
  cv::Mat frame;
  cap >> frame;
  std::cout << frame.size() << std::endl;

  while (1)
  {
    cap >> frame;
    int64 start = cv::getTickCount();
    std::vector<DetectionRectangle> detections = badacost.detect(frame);
    badacost.showResults(frame, detections);
    int64 end = cv::getTickCount();


    std::ostringstream buf;
    buf << "Mode: " << (useCPU ? "CPU" : "GPU") << " | "
        << "FPS: " << std::fixed << std::setprecision(1) << (cv::getTickFrequency() / (double)(end-start));
    putText(frame, buf.str(), cv::Point(10, frame.rows-20), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);

    cv::imshow("Webcam", frame);
    if (cv::waitKey(1) >= 0)
    {
      break;
    }

  }
  return 0;
}














