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

  /**
   * Computes the IoU of d.bbox with this->bbox rectangles.
   *
   * @param d rectangle detection to compare with.
   * @return The IoU score of d.bbox and this->bbox.
   */
  float
  overlap
    (
    const DetectionRectangle& d
    ) const;

  /**
   * Resize the bbox (without moving their centers).
   *
   * If wr>0 or hr>0, the w/h of each bbox is adjusted in the following order:
   *  if(hr~=0), h=h*hr; end
   *  if(wr~=0), w=w*wr; end
   *  if(hr==0), h=w/ar; end
   *  if(wr==0), w=h*ar; end
   * Only one of hr/wr may be set to 0, and then only if ar>0. If, however,
   *  hr=wr=0 and ar>0 then resizes bbs such that areas and centers are
   * preserved but aspect ratio becomes ar.
   *
   * @param hr ratio by which to multiply height (or 0)
   * @param wr ratio by which to multiply width (or 0)
   * @param aspect_ratio target aspect ratio (used only if hr=0 or wr=0)
   */
   void
   resize
     (
     float hr,
     float wr,
     float aspect_ratio = 0.0
     );

  /**
   * (Translated from P.Dollar toolbox).
   * Fix bbox aspect ratios (without moving the bbox center).
   *
   * The width (w) or height (h) of each bbox is adjusted so that w/h=ar (aspect_ratio).
   * The parameter flag controls whether w or h should change:
   *  flag==0: expand bb to given ar
   *  flag==1: shrink bb to given ar
   *  flag==2: use original w, alter h
   *  flag==3: use original h, alter w
   *  flag==4: preserve area, alter w and h
   * If ar==1 (the default), always converts bb to a square, hence the name.
   *
   * @param flag controls whether w or h should change
   * @param aspect_ratio desired aspect ratio
   */
  void
  squarify
    (
    int flag,
    float aspect_ratio = 1.0
    );

};

inline std::ostream&
operator<<
(
  std::ostream& os,
  const DetectionRectangle& d
)
{
  os << "[ x=" << d.bbox.x << ", y=" <<  d.bbox.y;
  os << ", w=" << d.bbox.width << ", h=" << d.bbox.height;
  os << ", score=" << d.score;
  os << ", class_index=" << d.class_index;
  os << " ] " << std::endl;

  return os;
}

inline std::ostream&
operator<<
(
  std::ostream& os,
  const std::vector<DetectionRectangle>& detections
)
{
  for(DetectionRectangle d: detections)
  {
    os << d;
  }
  os << "#detections = " << detections.size() << std::endl;

  return os;
}

/**
 * @brief nonMaximumSuppression Perform NMS over the detected bounding boxes.
 *
 * We implement the P.Dollar's toolbox bbNms with the following parameters:
 *
 *   - type: 'maxg'
 *   - overlap: 0.3
 *   - overDnm: 'union'
 *
 * @param dts
 * @param dts_nms
 */
void nonMaximumSuppression
  (
  std::vector<DetectionRectangle>& dts,
  std::vector<DetectionRectangle>& dts_nms
  );


#endif // DETECTION_HPP
