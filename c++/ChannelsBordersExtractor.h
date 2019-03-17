#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;

class ChannelsBordersExtractor{
	public:
		std::vector<cv::Mat> extractFeatures(cv::Mat image);
};

