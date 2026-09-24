#pragma once

#include "application/model.h"

namespace puzzle_demo {

Model GenerateRoundedRectangle(float left,
                               float top,
                               float right,
                               float bottom,
                               float arc_radius,
                               glm::vec3 color,
                               int precision = 8);

}  // namespace puzzle_demo
