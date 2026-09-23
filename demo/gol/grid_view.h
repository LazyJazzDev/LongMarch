#pragma once

#include <algorithm>
#include <cmath>

#include "glm/glm.hpp"

// Coordinates are framebuffer pixels; zoom 1 fits the whole grid.
struct GridView {
  float zoom{1.0f};
  glm::vec2 pan{0.0f};

  void Clamp(glm::vec2 viewport, glm::vec2 fitted_grid) {
    auto limit = glm::max((fitted_grid * zoom - viewport) * 0.5f, glm::vec2{0.0f});
    pan = glm::clamp(pan, -limit, limit);
  }

  void Zoom(float factor, glm::vec2 anchor_from_center) {
    const float next = std::clamp(zoom * factor, 1.0f, 12.0f);
    pan = anchor_from_center - (anchor_from_center - pan) * (next / zoom);
    zoom = next;
  }
};
