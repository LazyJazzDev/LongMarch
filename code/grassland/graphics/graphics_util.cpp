#include "grassland/graphics/graphics_util.h"
namespace grassland::graphics {
SamplerInfo::SamplerInfo()
    : min_filter(FILTER_MODE_LINEAR),
      mag_filter(FILTER_MODE_LINEAR),
      mip_filter(FILTER_MODE_LINEAR),
      address_mode_u(ADDRESS_MODE_REPEAT),
      address_mode_v(ADDRESS_MODE_REPEAT),
      address_mode_w(ADDRESS_MODE_REPEAT) {
}

SamplerInfo::SamplerInfo(FilterMode filter)
    : SamplerInfo(filter, ADDRESS_MODE_REPEAT) {
}

SamplerInfo::SamplerInfo(AddressMode address_mode)
    : SamplerInfo(FILTER_MODE_LINEAR, address_mode) {
}

SamplerInfo::SamplerInfo(FilterMode filter, AddressMode address_mode)
    : SamplerInfo(filter,
                  filter,
                  filter,
                  address_mode,
                  address_mode,
                  address_mode) {
}

SamplerInfo::SamplerInfo(FilterMode min_filter,
                         FilterMode mag_filter,
                         FilterMode mip_filter,
                         AddressMode address_mode_u,
                         AddressMode address_mode_v,
                         AddressMode address_mode_w)
    : min_filter(min_filter),
      mag_filter(mag_filter),
      mip_filter(mip_filter),
      address_mode_u(address_mode_u),
      address_mode_v(address_mode_v),
      address_mode_w(address_mode_w) {
}

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

uint32_t PixelSize(ImageFormat format) {
  switch (format) {
    case IMAGE_FORMAT_B8G8R8A8_UNORM:
      return 4;
    case IMAGE_FORMAT_R8G8B8A8_UNORM:
      return 4;
    case IMAGE_FORMAT_R32G32B32A32_SFLOAT:
      return 16;
    case IMAGE_FORMAT_R32G32B32_SFLOAT:
      return 12;
    case IMAGE_FORMAT_R32G32_SFLOAT:
      return 8;
    case IMAGE_FORMAT_R32_SFLOAT:
      return 4;
    case IMAGE_FORMAT_D32_SFLOAT:
      return 4;
    default:
      return 0;
  }
}
}  // namespace grassland::graphics
