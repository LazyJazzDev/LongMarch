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
} ImageFormat;

std::string to_string(ImageFormat value);
std::string ImageFormatName(ImageFormat value);

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

}  // namespace grassland::rhi
