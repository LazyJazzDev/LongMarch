#include "grassland/graphics/graphics_util.h"
namespace grassland::graphics {

const char *BackendAPIString(BackendAPI api) {
  switch (api) {
    case BACKEND_API_VULKAN:
      return "Vulkan";
    case BACKEND_API_D3D12:
      return "Direct3D 12";
    default:
      return "Unknown";
  }
}

bool SupportBackendAPI(BackendAPI api) {
  switch (api) {
#if LONGMARCH_D3D12_ENABLED
    case BACKEND_API_D3D12:
      return true;
#endif
#if LONGMARCH_VULKAN_ENABLED
    case BACKEND_API_VULKAN:
      return true;
#endif
    default:
      return false;
  }
}

SamplerInfo::SamplerInfo()
    : min_filter(FILTER_MODE_LINEAR),
      mag_filter(FILTER_MODE_LINEAR),
      mip_filter(FILTER_MODE_LINEAR),
      address_mode_u(ADDRESS_MODE_REPEAT),
      address_mode_v(ADDRESS_MODE_REPEAT),
      address_mode_w(ADDRESS_MODE_REPEAT) {
}

SamplerInfo::SamplerInfo(FilterMode filter) : SamplerInfo(filter, ADDRESS_MODE_REPEAT) {
}

SamplerInfo::SamplerInfo(AddressMode address_mode) : SamplerInfo(FILTER_MODE_LINEAR, address_mode) {
}

SamplerInfo::SamplerInfo(FilterMode filter, AddressMode address_mode)
    : SamplerInfo(filter, filter, filter, address_mode, address_mode, address_mode) {
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

BlendState::BlendState() {
  blend_enable = false;
  src_color = BLEND_FACTOR_ONE;
  dst_color = BLEND_FACTOR_ZERO;
  color_op = BLEND_OP_ADD;
  src_alpha = BLEND_FACTOR_ONE;
  dst_alpha = BLEND_FACTOR_ZERO;
  alpha_op = BLEND_OP_ADD;
}

BlendState::BlendState(bool blend_enable) : blend_enable(blend_enable) {
  src_color = BLEND_FACTOR_SRC_ALPHA;
  dst_color = BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  color_op = BLEND_OP_ADD;
  src_alpha = BLEND_FACTOR_ONE;
  dst_alpha = BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  alpha_op = BLEND_OP_ADD;
}

BlendState::BlendState(BlendFactor src_color,
                       BlendFactor dst_color,
                       BlendOp color_op,
                       BlendFactor src_alpha,
                       BlendFactor dst_alpha,
                       BlendOp alpha_op)
    : blend_enable(true),
      src_color(src_color),
      dst_color(dst_color),
      color_op(color_op),
      src_alpha(src_alpha),
      dst_alpha(dst_alpha),
      alpha_op(alpha_op) {
}

bool IsDepthFormat(ImageFormat format) {
  switch (format) {
    case IMAGE_FORMAT_D32_SFLOAT:
      return true;
    default:
      return false;
  }
}

glm::vec3 HSVtoRGB(const glm::vec3 hsv) {
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

float GreyScale(const glm::vec3 rgb) {
  return 0.299f * rgb.r + 0.587f * rgb.g + 0.114f * rgb.b;
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
