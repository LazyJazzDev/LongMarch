#pragma once
#include "grassland/vulkan/vulkan.h"

namespace grassland::graphics {

typedef enum BackendAPI {
  None = 0,
  Vulkan = 1,
  D3D12 = 2,
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

typedef enum AttributeFormat {
  UINT = 0,
  INT = 1,
  FLOAT = 2,
  UINT2 = 3,
  INT2 = 4,
  FLOAT2 = 5,
  UINT3 = 6,
  INT3 = 7,
  FLOAT3 = 8,
  UINT4 = 9,
  INT4 = 10,
  FLOAT4 = 11,
} AttributeFormat;

}  // namespace grassland::graphics
