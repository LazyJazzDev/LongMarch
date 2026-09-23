#include "util.h"

#include <iostream>

namespace geometry {

bool Between(float x0, float x1, float y) {
  if (x0 > x1)
    std::swap(x0, x1);
  return x0 < y && y < x1;
}
}  // namespace geometry
