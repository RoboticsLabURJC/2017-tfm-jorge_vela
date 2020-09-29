/** ------------------------------------------------------------------------
 *
 *  @brief DetectionRectangle.
 *  @author Jorge Vela
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2020/09/29
 *
 *  ------------------------------------------------------------------------ */
#ifndef DETECTION_RECTANGLE_HPP
#define DETECTION_RECTANGLE_HPP

#include <opencv2/opencv.hpp>

/** ------------------------------------------------------------------------
 *
 *  @brief Struct that represents a detected object using a Bounding Box.
 *
 *  ------------------------------------------------------------------------ */
struct DetectionRectangle
{
  cv::Rect bbox;   // Localization of the bounding box around the object.`
  float score;     // Detection score (the higher the more confident).
  int class_index; // Detected class index (e.g. representing car orientation)
};

#endif // DETECTION_HPP
