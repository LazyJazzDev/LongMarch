#pragma once
#include "GLFW/glfw3.h"
#include "glm/glm.hpp"
#include "grassland/util/util.h"

namespace grassland::graphics {

class Core;
class Buffer;
class Image;
class Sampler;
class Window;
class Shader;
class Program;
class CommandContext;

// Ray tracing objects
class RayTracingProgram;
class AccelerationStructure;

typedef enum BackendAPI {
  BACKEND_API_VULKAN = 0,
  BACKEND_API_D3D12 = 1,
  BACKEND_API_DEFAULT =
#ifdef _WIN32
      BACKEND_API_D3D12
#else
      BACKEND_API_VULKAN
#endif
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
  IMAGE_FORMAT_R16G16B16A16_SFLOAT = 8,
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
  RESOURCE_TYPE_SAMPLER = 4,
  RESOURCE_TYPE_ACCELERATION_STRUCTURE = 5,
} ResourceType;

typedef enum ShaderType {
  SHADER_TYPE_VERTEX = 0,
  SHADER_TYPE_FRAGMENT = 1,
  SHADER_TYPE_GEOMETRY = 2,
} ShaderType;

typedef enum BindPoint {
  BIND_POINT_GRAPHICS = 0,
  BIND_POINT_RAYTRACING = 1,
  BIND_POINT_COUNT = 2,
} BindPoint;

struct PhysicalDeviceProperties {
  std::string name;
  uint64_t score;
  bool ray_tracing_support;
  bool geometry_shader_support;
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

struct ClearValueUnion {
  ClearValue value;
  ClearValueUnion(ColorClearValue c) {
    value.color = c;
  }
  ClearValueUnion(DepthClearValue d) {
    value.depth = d;
  }
  explicit operator ClearValue() const {
    return value;
  }
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

typedef enum FilterMode {
  FILTER_MODE_NEAREST = 0,
  FILTER_MODE_LINEAR = 1,
} FilterMode;

typedef enum AddressMode {
  ADDRESS_MODE_REPEAT = 0,
  ADDRESS_MODE_MIRRORED_REPEAT = 1,
  ADDRESS_MODE_CLAMP_TO_EDGE = 2,
  ADDRESS_MODE_CLAMP_TO_BORDER = 3,
} AddressMode;

struct SamplerInfo {
  SamplerInfo();
  SamplerInfo(FilterMode filter);
  SamplerInfo(AddressMode address_mode);
  SamplerInfo(FilterMode filter, AddressMode address_mode);
  SamplerInfo(FilterMode min_filter,
              FilterMode mag_filter,
              FilterMode mip_filter,
              AddressMode address_mode_u,
              AddressMode address_mode_v,
              AddressMode address_mode_w);
  FilterMode min_filter;
  FilterMode mag_filter;
  FilterMode mip_filter;
  AddressMode address_mode_u;
  AddressMode address_mode_v;
  AddressMode address_mode_w;
};

typedef enum PrimitiveTopology {
  PRIMITIVE_TOPOLOGY_TRIANGLE_LIST = 0,
  PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP = 1,
  PRIMITIVE_TOPOLOGY_LINE_LIST = 2,
  PRIMITIVE_TOPOLOGY_LINE_STRIP = 3,
  PRIMITIVE_TOPOLOGY_POINT_LIST = 4,
} PrimitiveTopology;

typedef enum BlendFactor {
  BLEND_FACTOR_ZERO = 0,
  BLEND_FACTOR_ONE = 1,
  BLEND_FACTOR_SRC_COLOR = 2,
  BLEND_FACTOR_ONE_MINUS_SRC_COLOR = 3,
  BLEND_FACTOR_DST_COLOR = 4,
  BLEND_FACTOR_ONE_MINUS_DST_COLOR = 5,
  BLEND_FACTOR_SRC_ALPHA = 6,
  BLEND_FACTOR_ONE_MINUS_SRC_ALPHA = 7,
  BLEND_FACTOR_DST_ALPHA = 8,
  BLEND_FACTOR_ONE_MINUS_DST_ALPHA = 9,
} BlendFactor;

typedef enum BlendOp {
  BLEND_OP_ADD = 0,
  BLEND_OP_SUBTRACT = 1,
  BLEND_OP_REVERSE_SUBTRACT = 2,
  BLEND_OP_MIN = 3,
  BLEND_OP_MAX = 4,
} BlendOp;

typedef enum RayTracingInstanceFlag : uint32_t {
  RAYTRACING_INSTANCE_FLAG_NONE = 0,
  RAYTRACING_INSTANCE_FLAG_TRIANGLE_FACING_CULL_DISABLE = 0x00000001,
  RAYTRACING_INSTANCE_FLAG_TRIANGLE_FLIP_FACING = 0x00000002,
  RAYTRACING_INSTANCE_FLAG_OPAQUE = 0x00000004,
  RAYTRACING_INSTANCE_FLAG_NO_OPAQUE = 0x00000008
} RayTracingInstanceFlag;

struct BlendState {
  bool blend_enable;
  BlendFactor src_color;
  BlendFactor dst_color;
  BlendOp color_op;
  BlendFactor src_alpha;
  BlendFactor dst_alpha;
  BlendOp alpha_op;

  BlendState();

  BlendState(bool blend_enable);

  BlendState(BlendFactor src_color,
             BlendFactor dst_color,
             BlendOp color_op,
             BlendFactor src_alpha,
             BlendFactor dst_alpha,
             BlendOp alpha_op);
};

struct CompiledShaderBlob {
  std::vector<uint8_t> data;
  std::string entry_point;
};

struct RayTracingInstance {
  float transform[3][4];
  uint32_t instance_id : 24;
  uint32_t instance_mask : 8;
  uint32_t instance_hit_group_offset : 24;
  RayTracingInstanceFlag instance_flags : 8;
  AccelerationStructure *acceleration_structure;
};

#ifndef NDEBUG
constexpr bool kEnableDebug = true;
#else
constexpr bool kEnableDebug = false;
#endif

bool IsDepthFormat(ImageFormat format);

glm::vec3 HSVtoRGB(glm::vec3 hsv);

uint32_t PixelSize(ImageFormat format);

}  // namespace grassland::graphics
