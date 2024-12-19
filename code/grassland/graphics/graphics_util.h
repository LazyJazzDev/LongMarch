#pragma once
#include "GLFW/glfw3.h"
#include "glm/glm.hpp"
#include "grassland/util/util.h"

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

typedef enum ResourceType {
  RESOURCE_TYPE_UNIFORM_BUFFER = 0,
  RESOURCE_TYPE_STORAGE_BUFFER = 1,
  RESOURCE_TYPE_TEXTURE = 2,
  RESOURCE_TYPE_IMAGE = 3,
} ResourceType;

typedef enum ShaderType {
  SHADER_TYPE_VERTEX = 0,
  SHADER_TYPE_FRAGMENT = 1
} ShaderType;

struct PhysicalDeviceProperties {
  std::string name;
  uint64_t score;
  bool ray_tracing_support;
};

struct ColorClearValue {
  float r;
  float g;
  float b;
  float a;
};

struct DepthClearValue {
  float depth;
};

union ClearValue {
  ColorClearValue color;
  DepthClearValue depth;
};

struct Extent2D {
  uint32_t width;
  uint32_t height;
};

struct Offset2D {
  int x;
  int y;
};

struct Scissor {
  Offset2D offset;
  Extent2D extent;
};

struct Viewport {
  float x;
  float y;
  float width;
  float height;
  float min_depth;
  float max_depth;
};

typedef enum CullMode {
  CULL_MODE_NONE = 0,
  CULL_MODE_FRONT = 1,
  CULL_MODE_BACK = 2,
} CullMode;

class Core;
class Buffer;
class Image;
class Window;
class Shader;
class Program;
class CommandContext;

#ifndef NDEBUG
constexpr bool kEnableDebug = true;
#else
constexpr bool kEnableDebug = false;
#endif

bool IsDepthFormat(ImageFormat format);

glm::vec3 HSVtoRGB(glm::vec3 hsv);

}  // namespace grassland::graphics
