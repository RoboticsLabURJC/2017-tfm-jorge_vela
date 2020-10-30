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

float
DetectionRectangle::overlap
  (
  const DetectionRectangle& d
  ) const
{
  float w = std::min(bbox.width + bbox.x,
                     d.bbox.width + d.bbox.x)
            - std::max(bbox.x, d.bbox.x);

  if (w <= 0.0)
  {
    return 0.0;
  }

  float h = std::min(bbox.height + bbox.y,
                     d.bbox.height + d.bbox.y)
            - std::max(bbox.y, d.bbox.y);

  if (h <= 0.0)
  {
    return 0.0;
  }

  float i = w*h;
  float u = bbox.width * bbox.height + d.bbox.width * d.bbox.height - i;

  return i/u;
}

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


void nonMaximumSuppression
  (
  std::vector<DetectionRectangle>& dts,  // input
  std::vector<DetectionRectangle>& dts_nms // output
  )
{
  //float thr = -std::numeric_limits<float>::infinity();
  //float maxn = std::numeric_limits<float>::infinity(); // Maximum number of bboxes to output
  bool ovrDnm = true;
  float overlap = 0.3;
  std::vector<float> radii = {0.15, 0.15, 1., 1.};

  if (dts.size() == 0)
  {
    return;
  }

  // -----------------------------------------------------------
  // for each i suppress all j st j>i and area-overlap>overlap
  // -----------------------------------------------------------

  // Order dts rectangles by descending order of score.
  auto greater_score = [](DetectionRectangle& d1, DetectionRectangle& d2){ return d1.score > d2.score; };
  std::sort(dts.begin(), dts.end(), greater_score);

  std::vector<bool> kp(dts.size(), true);
  std::vector<int> areas;
  std::vector<int> x2, y2;
  for (DetectionRectangle d: dts)
  {
    areas.push_back(d.bbox.height * d.bbox.width);
    x2.push_back(d.bbox.x + d.bbox.width);
    y2.push_back(d.bbox.y + d.bbox.height);
  }

  for (uint i = 0; i < dts.size(); i++)
  {
    if (!kp[i])
    {
      continue;
    }
    for (uint j = (i+1); j <= dts.size(); j++)
    {
      if (!kp[j])
      {
        continue;
      }

      int iw = std::min(x2[i], x2[j]) - std::max(dts[i].bbox.x, dts[j].bbox.x);
      if (iw <= 0)
      {
        continue;
      }

      int ih = std::min(y2[i], y2[j]) - std::max(dts[i].bbox.y, dts[j].bbox.y);
      if (ih <= 0)
      {
        continue;
      }

      float o = iw*ih;
      int u;
      if (ovrDnm)
      {
        u = areas[i] + areas[j] - o;
      }
      else
      {
        u = std::min(areas[i], areas[j]);
      }
      o = o / static_cast<float>(u);
      if (o > overlap)
      {
        kp[j] = false;
      }
    }
  }

  for (uint i=0; i<dts.size(); i++)
  {
    if (kp[i])
    {
      dts_nms.push_back(dts[i]);
    }
  }
}



