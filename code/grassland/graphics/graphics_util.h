#pragma once
#include "grassland/vulkan/vulkan.h"

namespace grassland::graphics {

typedef enum BackendAPI {
  BACKEND_API_VULKAN = 0,
  BACKEND_API_D3D12 = 1,
} BackendAPI;

typedef enum ImageFormat {
  IMAGE_FORMAT_UNDEFINED = 0,
  IMAGE_FORMAT_B8G8R8A8_UNORM = 1,
  IMAGE_FORMAT_R8G8B8A8_UNORM = 2,
  IMAGE_FORMAT_R32G32B32A32_SFLOAT = 3,
  IMAGE_FORMAT_R32G32B32_SFLOAT = 4,
  IMAGE_FORMAT_R32G32_SFLOAT = 5,
  IMAGE_FORMAT_R32_SFLOAT = 6,
  IMAGE_FORMAT_D32_SFLOAT = 7,
} ImageFormat;

typedef enum InputType {
  INPUT_TYPE_UINT = 0,
  INPUT_TYPE_INT = 1,
  INPUT_TYPE_FLOAT = 2,
  INPUT_TYPE_UINT2 = 3,
  INPUT_TYPE_INT2 = 4,
  INPUT_TYPE_FLOAT2 = 5,
  INPUT_TYPE_UINT3 = 6,
  INPUT_TYPE_INT3 = 7,
  INPUT_TYPE_FLOAT3 = 8,
  INPUT_TYPE_UINT4 = 9,
  INPUT_TYPE_INT4 = 10,
  INPUT_TYPE_FLOAT4 = 11,
} InputType;

typedef enum BufferType {
  BUFFER_TYPE_STATIC = 0,
  BUFFER_TYPE_DYNAMIC = 1,
  BUFFER_TYPE_ONETIME = 2,
} BufferType;

struct PhysicalDeviceProperties {
  std::string name;
  uint64_t score;
  bool ray_tracing_support;
};

class Core;
class Buffer;
class Image;

#ifndef NDEBUG
constexpr bool kEnableDebug = true;
#else
constexpr bool kEnableDebug = false;
#endif

}  // namespace grassland::graphics
