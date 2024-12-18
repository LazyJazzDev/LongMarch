#include "grassland/graphics/graphics_util.h"
namespace grassland::graphics {
bool IsDepthFormat(ImageFormat format) {
  switch (format) {
    case IMAGE_FORMAT_D32_SFLOAT:
      return true;
    default:
      return false;
  }
}

glm::vec3 HSVtoRGB(glm::vec3 hsv) {
  float c = hsv.z * hsv.y;
  float h = hsv.x * 360.0 / 60.0f;
  float x = c * (1.0f - std::abs(std::fmod(h, 2.0f) - 1.0f));
  float m = hsv.z - c;
  glm::vec3 rgb;
  if (h >= 0.0f && h < 1.0f) {
    rgb = glm::vec3(c, x, 0.0f);
  } else if (h >= 1.0f && h < 2.0f) {
    rgb = glm::vec3(x, c, 0.0f);
  } else if (h >= 2.0f && h < 3.0f) {
    rgb = glm::vec3(0.0f, c, x);
  } else if (h >= 3.0f && h < 4.0f) {
    rgb = glm::vec3(0.0f, x, c);
  } else if (h >= 4.0f && h < 5.0f) {
    rgb = glm::vec3(x, 0.0f, c);
  } else if (h >= 5.0f && h < 6.0f) {
    rgb = glm::vec3(c, 0.0f, x);
  }
  return rgb + glm::vec3(m);
}
}  // namespace grassland::graphics
