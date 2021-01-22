#include <opencv2/opencv.hpp>
#include <chrono>

using namespace cv;

int main(int argc, char** argv)
{
    Mat img1 = imread("obj_detect/tests/images/coches10.jpg", IMREAD_COLOR);
    resize(img1, img1, Size(0,0), 5, 5, cv::INTER_LINEAR);

    auto start = std::chrono::system_clock::now();
    Mat img, gray;
    img1.copyTo(img);

    cvtColor(img, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, gray,Size(7, 7), 1.5);
    Canny(gray, gray, 0, 50);

    gray.copyTo(gray);

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<float,std::milli> duration = end - start;
    std::cout << duration.count() << "ms" << std::endl;

//    imshow("edges", gray);
//    waitKey();
    return 0;
}
