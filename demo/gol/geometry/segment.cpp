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

bool Segment::operator<(const Segment &segment) const {
#define COMPARE(x)   \
  if (x < segment.x) \
    return true;     \
  if (segment.x < x) \
    return false;
  COMPARE(u);
  COMPARE(v);
#undef COMPARE
  return false;
}

bool Segment::IsPoint() const {
  return length(u - v) < eps;
}
}  // namespace geometry
