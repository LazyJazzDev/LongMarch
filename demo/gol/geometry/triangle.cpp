#include "triangle.h"

#include "line.h"
#include "segment.h"
#include "util.h"

namespace geometry {
using namespace glm;
#ifdef COMPARE
#undef COMPARE
#endif
namespace {
bool operator<(const vec2 &v0, const vec2 &v1) {
#define COMPARE(x) \
  if (v0.x < v1.x) \
    return true;   \
  if (v1.x < v0.x) \
    return false;
  COMPARE(x);
  COMPARE(y);
#undef COMPARE
  return false;
}
}  // namespace

bool Triangle::operator<(const Triangle &triangle) const {
#define COMPARE(x)    \
  if (x < triangle.x) \
    return true;      \
  if (triangle.x < x) \
    return false;
  COMPARE(v0);
  COMPARE(v1);
  COMPARE(v2);
#undef COMPARE
  return false;
}

bool Triangle::IsInside(glm::vec2 v) const {
  auto signal = [](float x) {
    if (x < 0.0)
      return -1;
    if (x > 0.0)
      return 1;
    return 0;
  };
  auto line2 = Line{Segment{v0, v1}};
  auto line0 = Line{Segment{v1, v2}};
  auto line1 = Line{Segment{v2, v0}};
  return (std::abs(signal(line0(v0)) - signal(line0(v))) < 2) && (std::abs(signal(line1(v1)) - signal(line1(v))) < 2) &&
         (std::abs(signal(line2(v2)) - signal(line2(v))) < 2) && (v != v0 && v != v1 && v != v2);
}
}  // namespace geometry
