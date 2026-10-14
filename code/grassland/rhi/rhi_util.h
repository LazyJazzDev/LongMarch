#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace grassland::rhi {

void ReportError(int32_t error_code, const std::string &error_message);

class IObject;

namespace backend {
struct Access;
class ICore;
class IDevice;
class IWindow;
class IImage;
class IQueue;
class ICommandContext;
}  // namespace backend

class ObjectRef {
 public:
  ObjectRef() noexcept;
  ObjectRef(const ObjectRef &other) noexcept;
  ObjectRef(ObjectRef &&other) noexcept;
  ObjectRef &operator=(const ObjectRef &other) noexcept;
  ObjectRef &operator=(ObjectRef &&other) noexcept;
  ~ObjectRef();
  explicit operator bool() const noexcept;

 private:
  IObject *obj_{nullptr};
  friend struct backend::Access;
};

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

class Core;
class Device;

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

enum class PresentMode { VSync, Continuous };
enum class QueueType { Graphics, Transfer };
enum class SubmissionStatus { Pending, Completed, Failed, Canceled };
enum class ImageFormat { Unknown, RGBA8Unorm, BGRA8Unorm };
enum class Key { P, Escape };

struct WindowDesc {
  int width{1280};
  int height{720};
  std::string title{"LongMarch RHI"};
  PresentMode present_mode{PresentMode::VSync};
};

struct ImageDesc {
  uint32_t width{};
  uint32_t height{};
  ImageFormat format{ImageFormat::Unknown};
};

}  // namespace grassland::rhi
