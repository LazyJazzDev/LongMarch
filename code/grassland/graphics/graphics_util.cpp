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
#if defined(LONGMARCH_D3D12_ENABLED)
    case BACKEND_API_D3D12:
      return true;
#endif
#if defined(LONGMARCH_VULKAN_ENABLED)
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

void util::PybindModuleRegistration(py::module_ &m) {
  py::enum_<BackendAPI> backend_api(m, "BackendAPI");
  backend_api.value("BACKEND_API_VULKAN", BACKEND_API_VULKAN, "Backend API: Vulkan");
  backend_api.value("BACKEND_API_D3D12", BACKEND_API_D3D12, "Backend API: Direct3D 12");
  backend_api.export_values();
  m.attr("BACKEND_API_DEFAULT") = py::cast(BACKEND_API_DEFAULT);

  py::enum_<ImageFormat> image_format(m, "ImageFormat");
  image_format.value("IMAGE_FORMAT_UNDEFINED", IMAGE_FORMAT_UNDEFINED, "Image Format: Undefined");
  image_format.value("IMAGE_FORMAT_B8G8R8A8_UNORM", IMAGE_FORMAT_B8G8R8A8_UNORM, "Image Format: B8G8R8A8 Unorm");
  image_format.value("IMAGE_FORMAT_R8G8B8A8_UNORM", IMAGE_FORMAT_R8G8B8A8_UNORM, "Image Format: R8G8B8A8 Unorm");
  image_format.value("IMAGE_FORMAT_R32G32B32A32_SFLOAT", IMAGE_FORMAT_R32G32B32A32_SFLOAT,
                     "Image Format: R32G32B32A32 Float");
  image_format.value("IMAGE_FORMAT_R32G32B32_SFLOAT", IMAGE_FORMAT_R32G32B32_SFLOAT, "Image Format: R32G32B32 Float");
  image_format.value("IMAGE_FORMAT_R32G32_SFLOAT", IMAGE_FORMAT_R32G32_SFLOAT, "Image Format: R32G32 Float");
  image_format.value("IMAGE_FORMAT_R32_SFLOAT", IMAGE_FORMAT_R32_SFLOAT, "Image Format: R32 Float");
  image_format.value("IMAGE_FORMAT_D32_SFLOAT", IMAGE_FORMAT_D32_SFLOAT, "Image Format: D32 Float");
  image_format.value("IMAGE_FORMAT_R16G16B16A16_SFLOAT", IMAGE_FORMAT_R16G16B16A16_SFLOAT,
                     "Image Format: R16G16B16A16 Float");
  image_format.value("IMAGE_FORMAT_R32_UINT", IMAGE_FORMAT_R32_UINT, "Image Format: R32 Uint");
  image_format.value("IMAGE_FORMAT_R32_SINT", IMAGE_FORMAT_R32_SINT, "Image Format: R32 Sint");
  image_format.export_values();

  py::enum_<InputType> input_type(m, "InputType");
  input_type.value("INPUT_TYPE_UINT", INPUT_TYPE_UINT, "Input Type: uint");
  input_type.value("INPUT_TYPE_INT", INPUT_TYPE_INT, "Input Type: int");
  input_type.value("INPUT_TYPE_FLOAT", INPUT_TYPE_FLOAT, "Input Type: float");
  input_type.value("INPUT_TYPE_UINT2", INPUT_TYPE_UINT2, "Input Type: uint2");
  input_type.value("INPUT_TYPE_INT2", INPUT_TYPE_INT2, "Input Type: int2");
  input_type.value("INPUT_TYPE_FLOAT2", INPUT_TYPE_FLOAT2, "Input Type: float2");
  input_type.value("INPUT_TYPE_UINT3", INPUT_TYPE_UINT3, "Input Type: uint3");
  input_type.value("INPUT_TYPE_INT3", INPUT_TYPE_INT3, "Input Type: int3");
  input_type.value("INPUT_TYPE_FLOAT3", INPUT_TYPE_FLOAT3, "Input Type: float3");
  input_type.value("INPUT_TYPE_UINT4", INPUT_TYPE_UINT4, "Input Type: uint4");
  input_type.value("INPUT_TYPE_INT4", INPUT_TYPE_INT4, "Input Type: int4");
  input_type.value("INPUT_TYPE_FLOAT4", INPUT_TYPE_FLOAT4, "Input Type: float4");
  input_type.export_values();

  py::enum_<BufferType> buffer_type(m, "BufferType");
  buffer_type.value("BUFFER_TYPE_STATIC", BUFFER_TYPE_STATIC, "Buffer Type: Static");
  buffer_type.value("BUFFER_TYPE_DYNAMIC", BUFFER_TYPE_DYNAMIC, "Buffer Type: Dynamic");
  buffer_type.value("BUFFER_TYPE_ONETIME", BUFFER_TYPE_ONETIME, "Buffer Type: One-Time");
  buffer_type.export_values();

  py::enum_<ResourceType> resource_type(m, "ResourceType");
  resource_type.value("RESOURCE_TYPE_UNIFORM_BUFFER", RESOURCE_TYPE_UNIFORM_BUFFER, "Resource Type: Uniform Buffer");
  resource_type.value("RESOURCE_TYPE_STORAGE_BUFFER", RESOURCE_TYPE_STORAGE_BUFFER, "Resource Type: Storage Buffer");
  resource_type.value("RESOURCE_TYPE_IMAGE", RESOURCE_TYPE_IMAGE, "Resource Type: Image (read-only)");
  resource_type.value("RESOURCE_TYPE_WRITABLE_IMAGE", RESOURCE_TYPE_WRITABLE_IMAGE, "Resource Type: Writable Image");
  resource_type.value("RESOURCE_TYPE_SAMPLER", RESOURCE_TYPE_SAMPLER, "Resource Type: Sampler");
  resource_type.value("RESOURCE_TYPE_ACCELERATION_STRUCTURE", RESOURCE_TYPE_ACCELERATION_STRUCTURE,
                      "Resource Type: Acceleration Structure");
  resource_type.value("RESOURCE_TYPE_WRITABLE_STORAGE_BUFFER", RESOURCE_TYPE_WRITABLE_STORAGE_BUFFER,
                      "Resource Type: Writable Storage Buffer");
  resource_type.export_values();

  py::enum_<ShaderType> shader_type(m, "ShaderType");
  shader_type.value("SHADER_TYPE_VERTEX", SHADER_TYPE_VERTEX, "Shader Type: Vertex Shader");
  shader_type.value("SHADER_TYPE_FRAGMENT", SHADER_TYPE_FRAGMENT, "Shader Type: Fragment Shader");
  shader_type.value("SHADER_TYPE_GEOMETRY", SHADER_TYPE_GEOMETRY, "Shader Type: Geometry Shader");
  shader_type.export_values();

  py::enum_<BindPoint> bind_point(m, "BindPoint");
  bind_point.value("BIND_POINT_GRAPHICS", BIND_POINT_GRAPHICS, "Bind Point: Graphics Pipeline");
  bind_point.value("BIND_POINT_COMPUTE", BIND_POINT_COMPUTE, "Bind Point: Compute Pipeline");
  bind_point.value("BIND_POINT_RAYTRACING", BIND_POINT_RAYTRACING, "Bind Point: Ray Tracing Pipeline");
}

}  // namespace grassland::graphics
