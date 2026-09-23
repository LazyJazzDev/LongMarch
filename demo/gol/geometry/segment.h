#pragma once
#include <glm/glm.hpp>

namespace geometry {

struct Segment {
  glm::vec2 u;
  glm::vec2 v;
  [[nodiscard]] bool IsPoint() const;
  bool operator<(const Segment &segment) const;
};

}  // namespace geometry
