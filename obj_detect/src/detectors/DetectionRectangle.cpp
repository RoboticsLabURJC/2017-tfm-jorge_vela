/** ------------------------------------------------------------------------
 *
 *  @brief DetectionRectangle.
 *  @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
 *  @date 2020/09/29
 *
 *  ------------------------------------------------------------------------ */

#include <detectors/DetectionRectangle.h>
#include <opencv2/opencv.hpp>
#include <cmath>

void
DetectionRectangle::resize
  (
  float hr,
  float wr,
  float aspect_ratio
  )
{
  assert( ((hr > 0) && (wr > 0)) || aspect_ratio > 0.0);

  // preserve area and center, set aspect ratio
  if ((hr==0) && (wr==0))
  {
    float area = sqrt(bbox.width * bbox.height);
    float ar = sqrt(aspect_ratio);
    int d = round(area*ar) - bbox.width;
    bbox.x -= round(d/2.0);
    bbox.width += d;
    d = area/ar - bbox.height;
    bbox.y -= round(d/2.0);
    bbox.width += d;
  }
  else
  {
    // possibly adjust h/w based on hr/wr
    if (hr != 0)
    {
      int d = (hr - 1) * bbox.height;
      bbox.y -= round(d/2.0);
      bbox.height += d;
    }

    if (wr != 0)
    {
      int d = (wr-1) * bbox.width;
      bbox.x -= round(d/2.0);
      bbox.width += d;
    }

    // possibly adjust h/w based on ar and NEW h/w
    if (!hr)
    {
      int d = round(bbox.width/aspect_ratio) - bbox.height;
      bbox.y -= round(d/2.0);
      bbox.height += d;
    }

    if (!wr)
    {
      int d = round(bbox.height*aspect_ratio - bbox.width);
      bbox.x -= round(d/2.0);
      bbox.width += d;
    }
  }
}

void
DetectionRectangle::squarify
  (
  int flag,
  float aspect_ratio
  )
{
  if (flag == 4)
  {
    this->resize(0, 0, aspect_ratio);
    return;
  }

  bool usew = ((flag==0) && (bbox.width > bbox.height*aspect_ratio));
  usew = usew || ((flag == 1) && (bbox.width < bbox.height*aspect_ratio));
  usew = usew || (flag==2);
  if (usew)
  {
    this->resize(0, 1, aspect_ratio);
  }
  else
  {
    this->resize(1, 0, aspect_ratio);
  }
};



