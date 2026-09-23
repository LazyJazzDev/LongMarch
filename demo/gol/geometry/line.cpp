#include "line.h"

#include "util.h"

namespace geometry {
using namespace glm;

Line::Line(const Segment &segment) : n(segment.v - segment.u) {
  std::swap(n.x, n.y);
  n.x = -n.x;
  n /= glm::length(n);
  if (n.x < 0.0 || (n.x == 0 && n.y < 0.0))
    n = -n;
  d = -glm::dot(n, segment.v);
}

bool Line::operator==(const Line &line) const {
  return Equal(n, line.n) && Equal(d, line.d);
}

float Line::operator()(const vec2 &v) const {
  return dot(v, n) + d;
}

glm::vec2 Intersection(const Line &line0, const Line &line1) {
  return -vec2(line0.d, line1.d) * inverse(mat2{line0.n.x, line0.n.y, line1.n.x, line1.n.y});
}
}  // namespace geometry
