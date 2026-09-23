#pragma once

#include <glm/glm.hpp>

namespace geometry {
struct Triangle {
  glm::vec2 v0;
  glm::vec2 v1;
  glm::vec2 v2;
  bool operator<(const Triangle &triangle) const;
  [[nodiscard]] bool IsInside(glm::vec2 v) const;
};
}  // namespace geometry
