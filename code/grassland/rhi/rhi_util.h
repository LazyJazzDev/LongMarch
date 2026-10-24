#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace grassland::rhi {

typedef enum {
  BACKEND_API_D3D12 = 0,
  BACKEND_API_VULKAN = 1,
  BACKEND_API_METAL = 2,
  BACKEND_API_DEFAULT =

// Windows: D3D12 Apple(macOS, iOS): Metal, Linux: Vulkan
#if defined(LONGMARCH_D3D12_ENABLED)
      BACKEND_API_D3D12
#elif defined(LONGMARCH_METAL_ENABLED)
      BACKEND_API_METAL
#else
      BACKEND_API_VULKAN
#endif
} BackendAPI;

std::string to_string(BackendAPI api);
std::string BackendAPIName(BackendAPI api);

typedef enum {
  PHYSICAL_DEVICE_TYPE_INTEGRATED = 0,
  PHYSICAL_DEVICE_TYPE_DEDICATED = 1,
  PHYSICAL_DEVICE_TYPE_SOFTWARE = 2,
  PHYSICAL_DEVICE_TYPE_OTHER = 3,
} PhysicalDeviceType;

std::string to_string(PhysicalDeviceType type);
std::string PhysicalDeviceTypeName(PhysicalDeviceType type);

struct PhysicalDevice {
  std::string name;
  // Bytes of host/shared memory accessible to the GPU (not additional VRAM).
  uint64_t unified_memory{};
  uint64_t dedicated_memory{};
  PhysicalDeviceType type{PHYSICAL_DEVICE_TYPE_INTEGRATED};
  bool ray_tracing_pipeline_feature{};
  bool ray_query_feature{};
  bool geometry_shader_feature{};
};

typedef enum {
  PRESENT_MODE_VSYNC,
  PRESENT_MODE_CONTINUOUS,
} PresentMode;

std::string to_string(PresentMode value);
std::string PresentModeName(PresentMode value);

typedef enum {
  QUEUE_TYPE_GRAPHICS,
  QUEUE_TYPE_TRANSFER,
} QueueType;

std::string to_string(QueueType value);
std::string QueueTypeName(QueueType value);

typedef enum {
  SUBMISSION_STATUS_PENDING,
  SUBMISSION_STATUS_COMPLETED,
  SUBMISSION_STATUS_FAILED,
  SUBMISSION_STATUS_CANCELED,
} SubmissionStatus;

std::string to_string(SubmissionStatus value);
std::string SubmissionStatusName(SubmissionStatus value);

typedef enum {
  IMAGE_FORMAT_UNKNOWN,
  IMAGE_FORMAT_R8_UINT,
  IMAGE_FORMAT_R8_INT,
  IMAGE_FORMAT_R8_UNORM,
  IMAGE_FORMAT_R8G8_UINT,
  IMAGE_FORMAT_R8G8_INT,
  IMAGE_FORMAT_R8G8_UNORM,
  IMAGE_FORMAT_R8G8B8_UINT,
  IMAGE_FORMAT_R8G8B8_INT,
  IMAGE_FORMAT_R8G8B8_UNORM,
  IMAGE_FORMAT_R8G8B8A8_UINT,
  IMAGE_FORMAT_R8G8B8A8_INT,
  IMAGE_FORMAT_R8G8B8A8_UNORM,
  IMAGE_FORMAT_R32_UINT,
  IMAGE_FORMAT_R32_INT,
  IMAGE_FORMAT_R32_SFLOAT,
  IMAGE_FORMAT_R32G32_UINT,
  IMAGE_FORMAT_R32G32_INT,
  IMAGE_FORMAT_R32G32_SFLOAT,
  IMAGE_FORMAT_R32G32B32_UINT,
  IMAGE_FORMAT_R32G32B32_INT,
  IMAGE_FORMAT_R32G32B32_SFLOAT,
  IMAGE_FORMAT_R32G32B32A32_UINT,
  IMAGE_FORMAT_R32G32B32A32_INT,
  IMAGE_FORMAT_R32G32B32A32_SFLOAT,
  IMAGE_FORMAT_B8G8R8A8_UNORM,
  IMAGE_FORMAT_R16G16B16A16_SFLOAT,
  IMAGE_FORMAT_R10G10B10A2_UNORM,
  IMAGE_FORMAT_B10G10R10A2_UNORM,
  IMAGE_FORMAT_D16_UNORM,
  IMAGE_FORMAT_D32_SFLOAT,
  IMAGE_FORMAT_D24_UNORM_S8_UINT,
} ImageFormat;

std::string to_string(ImageFormat value);
std::string ImageFormatName(ImageFormat value);
bool IsDepthFormat(ImageFormat format);
bool HasStencil(ImageFormat format);
uint32_t ImageFormatSize(ImageFormat format);
bool IsColorFormat(ImageFormat format);
bool IsSignedIntegerFormat(ImageFormat format);
bool IsUnsignedIntegerFormat(ImageFormat format);

enum CompareOp {
  COMPARE_NEVER,
  COMPARE_LESS,
  COMPARE_EQUAL,
  COMPARE_LESS_EQUAL,
  COMPARE_GREATER,
  COMPARE_NOT_EQUAL,
  COMPARE_GREATER_EQUAL,
  COMPARE_ALWAYS
};

enum StencilOp {
  STENCIL_KEEP,
  STENCIL_ZERO,
  STENCIL_REPLACE,
  STENCIL_INCREMENT_CLAMP,
  STENCIL_DECREMENT_CLAMP,
  STENCIL_INVERT,
  STENCIL_INCREMENT_WRAP,
  STENCIL_DECREMENT_WRAP
};

struct StencilFaceDesc {
  StencilOp fail_op{STENCIL_KEEP}, depth_fail_op{STENCIL_KEEP}, pass_op{STENCIL_KEEP};
  CompareOp compare{COMPARE_ALWAYS};
};

struct DepthStencilDesc {
  ImageFormat format{IMAGE_FORMAT_UNKNOWN};
  bool depth_test{}, depth_write{}, stencil_test{};
  CompareOp depth_compare{COMPARE_LESS};
  uint8_t stencil_read_mask{0xff}, stencil_write_mask{0xff};
  uint32_t stencil_reference{};
  StencilFaceDesc front, back;
};

// Values written to a swapchain must use its declared encoding. No implicit
// tonemapping or color conversion is performed by Present.
enum ColorSpace {
  COLOR_SPACE_SRGB_NONLINEAR,
  // Linear Rec.709/scRGB: 1.0 corresponds to 80 cd/m^2; values may exceed 1.
  COLOR_SPACE_EXTENDED_SRGB_LINEAR,
  // Rec.2020 primaries, ST.2084/PQ encoded absolute luminance (0..10000 cd/m^2).
  COLOR_SPACE_HDR10_ST2084,
  // Linear Rec.709 with platform-defined luminance mapping; not the scRGB contract.
  COLOR_SPACE_BT709_LINEAR,
};

std::string to_string(ColorSpace value);
std::string ColorSpaceName(ColorSpace value);

struct WindowPresentationOption {
  PresentMode present_mode;
  ImageFormat format;
  ColorSpace color_space{COLOR_SPACE_SRGB_NONLINEAR};
};

struct WindowDesc {
  int width{1280};
  int height{720};
  std::string title{"LongMarch RHI"};
  PresentMode present_mode{PRESENT_MODE_VSYNC};
  ColorSpace color_space{COLOR_SPACE_SRGB_NONLINEAR};
};

struct ImageDesc {
  uint32_t width{};
  uint32_t height{};
  ImageFormat format{IMAGE_FORMAT_UNKNOWN};
};

typedef enum {
  SHADER_STAGE_VERTEX,
  SHADER_STAGE_FRAGMENT,
  SHADER_STAGE_COMPUTE,
  SHADER_STAGE_RAY_GENERATION,
  SHADER_STAGE_MISS,
  SHADER_STAGE_CLOSEST_HIT,
  SHADER_STAGE_ANY_HIT,
  SHADER_STAGE_INTERSECTION,
  SHADER_STAGE_CALLABLE,
} ShaderStage;

std::string to_string(ShaderStage stage);

typedef enum {
  INPUT_TYPE_FLOAT,
  INPUT_TYPE_FLOAT2,
  INPUT_TYPE_FLOAT3,
  INPUT_TYPE_FLOAT4,
  INPUT_TYPE_INT,
  INPUT_TYPE_INT2,
  INPUT_TYPE_INT3,
  INPUT_TYPE_INT4,
  INPUT_TYPE_UINT,
  INPUT_TYPE_UINT2,
  INPUT_TYPE_UINT3,
  INPUT_TYPE_UINT4,
} InputType;

std::string to_string(InputType format);
uint32_t InputTypeSize(InputType format);

typedef enum { ATTACHMENT_LOAD_OP_LOAD, ATTACHMENT_LOAD_OP_CLEAR, ATTACHMENT_LOAD_OP_DISCARD } AttachmentLoadOp;

typedef enum { ATTACHMENT_STORE_OP_STORE, ATTACHMENT_STORE_OP_DISCARD } AttachmentStoreOp;

std::string to_string(AttachmentLoadOp op);
std::string to_string(AttachmentStoreOp op);

typedef enum { INDEX_FORMAT_UINT16, INDEX_FORMAT_UINT32 } IndexFormat;

std::string to_string(IndexFormat format);

typedef enum {
  BLEND_FACTOR_ZERO,
  BLEND_FACTOR_ONE,
  BLEND_FACTOR_SRC_COLOR,
  BLEND_FACTOR_ONE_MINUS_SRC_COLOR,
  BLEND_FACTOR_DST_COLOR,
  BLEND_FACTOR_ONE_MINUS_DST_COLOR,
  BLEND_FACTOR_SRC_ALPHA,
  BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
  BLEND_FACTOR_DST_ALPHA,
  BLEND_FACTOR_ONE_MINUS_DST_ALPHA,
} BlendFactor;

typedef enum {
  BLEND_OP_ADD,
  BLEND_OP_SUBTRACT,
  BLEND_OP_REVERSE_SUBTRACT,
  BLEND_OP_MIN,
  BLEND_OP_MAX,
} BlendOp;

struct BlendState {
  bool blend_enable;
  BlendFactor src_color, dst_color;
  BlendOp color_op;
  BlendFactor src_alpha, dst_alpha;
  BlendOp alpha_op;

  BlendState();
  // Standard source-alpha blending, including source-over alpha accumulation.
  BlendState(bool enabled);
  BlendState(BlendFactor src_color,
             BlendFactor dst_color,
             BlendOp color_op,
             BlendFactor src_alpha,
             BlendFactor dst_alpha,
             BlendOp alpha_op);
};

std::string to_string(BlendFactor value);
std::string to_string(BlendOp value);

typedef enum {
  ACCELERATION_STRUCTURE_BOTTOM_LEVEL,
  ACCELERATION_STRUCTURE_TOP_LEVEL,
} AccelerationStructureType;

typedef enum {
  ACCELERATION_STRUCTURE_BUILD,
  ACCELERATION_STRUCTURE_UPDATE,
} AccelerationStructureBuildMode;

std::string to_string(AccelerationStructureType value);
std::string to_string(AccelerationStructureBuildMode value);

}  // namespace grassland::rhi
