#include "rounded_rectangle.h"

#include "application//interpolation.h"

Model GenerateRoundedRectangle(float left,
                               float top,
                               float right,
                               float bottom,
                               float arc_radius,
                               glm::vec3 color,
                               int precision) {
  if (top > bottom)
    std::swap(top, bottom);
  arc_radius = std::min(arc_radius, std::min(bottom - top, right - left) * 0.5f);
  std::vector<Vertex> vertices;
  std::vector<uint32_t> indices;

  auto add_src = [&vertices, &color, &arc_radius](glm::vec2 origin, float theta_beg, float theta_end, int precision) {
    for (int i = 0; i <= precision; i++) {
      float alpha = float(i) / float(precision);
      float theta = Mix(theta_beg, theta_end, alpha);
      float sin_t = std::sin(theta), cos_t = std::cos(theta);
      vertices.push_back({{origin + glm::vec2{cos_t, sin_t} * arc_radius}, glm::vec4{color, 1.0f}});
    }
  };

  vertices.push_back({{(left + right) * 0.5f, (top + bottom) * 0.5f}, glm::vec4{color, 1.0f}});
  add_src({right - arc_radius, bottom - arc_radius}, glm::radians(0.0f), glm::radians(90.0f), precision);
  add_src({left + arc_radius, bottom - arc_radius}, glm::radians(90.0f), glm::radians(180.0f), precision);
  add_src({left + arc_radius, top + arc_radius}, glm::radians(180.0f), glm::radians(270.0f), precision);
  add_src({right - arc_radius, top + arc_radius}, glm::radians(270.0f), glm::radians(360.0f), precision);
  for (int i = 0; i < (precision + 1) * 4; i++) {
    int i1 = (i + 1) % ((precision + 1) * 4);
    indices.push_back(0);
    indices.push_back(i + 1);
    indices.push_back(i1 + 1);
  }

  return Model{vertices, indices};
}
