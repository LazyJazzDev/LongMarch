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
  IMAGE_FORMAT_RGBA8_UNORM,
  IMAGE_FORMAT_BGRA8_UNORM,
  IMAGE_FORMAT_D32_FLOAT,
  IMAGE_FORMAT_D24_UNORM_S8_UINT,
} ImageFormat;

std::string to_string(ImageFormat value);
std::string ImageFormatName(ImageFormat value);
bool IsDepthFormat(ImageFormat format);
bool HasStencil(ImageFormat format);

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

struct WindowPresentationOption {
  PresentMode present_mode;
  ImageFormat format;
};

struct WindowDesc {
  int width{1280};
  int height{720};
  std::string title{"LongMarch RHI"};
  PresentMode present_mode{PRESENT_MODE_VSYNC};
};

struct ImageDesc {
  uint32_t width{};
  uint32_t height{};
  ImageFormat format{IMAGE_FORMAT_UNKNOWN};
};

typedef enum { SHADER_STAGE_VERTEX, SHADER_STAGE_FRAGMENT } ShaderStage;

std::string to_string(ShaderStage stage);

typedef enum { VERTEX_FORMAT_FLOAT, VERTEX_FORMAT_FLOAT2, VERTEX_FORMAT_FLOAT3, VERTEX_FORMAT_FLOAT4 } VertexFormat;

std::string to_string(VertexFormat format);
uint32_t VertexFormatSize(VertexFormat format);

typedef enum { ATTACHMENT_LOAD_OP_LOAD, ATTACHMENT_LOAD_OP_CLEAR, ATTACHMENT_LOAD_OP_DISCARD } AttachmentLoadOp;

typedef enum { ATTACHMENT_STORE_OP_STORE, ATTACHMENT_STORE_OP_DISCARD } AttachmentStoreOp;

std::string to_string(AttachmentLoadOp op);
std::string to_string(AttachmentStoreOp op);

typedef enum { INDEX_FORMAT_UINT16, INDEX_FORMAT_UINT32 } IndexFormat;

std::string to_string(IndexFormat format);

}  // namespace grassland::rhi
