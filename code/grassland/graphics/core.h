#pragma once
#include "grassland/graphics/graphics_util.h"

namespace grassland::graphics {

class Core {
 public:
  struct Settings {
    int frames_in_flight{2};
    bool enable_debug{kEnableDebug};
  };
  Core(const Settings &settings);

  virtual ~Core() = default;

  virtual int CreateBuffer(size_t size,
                           BufferType type,
                           double_ptr<Buffer> pp_buffer) = 0;

  virtual int CreateImage(int width,
                          int height,
                          ImageFormat format,
                          double_ptr<Image> pp_image) = 0;

  virtual int CreateWindowObject(int width,
                                 int height,
                                 const std::string &title,
                                 double_ptr<Window> pp_window) = 0;

  virtual int GetPhysicalDeviceProperties(
      PhysicalDeviceProperties *p_physical_device_properties = nullptr) = 0;

  virtual int InitializeLogicalDevice(int device_index) = 0;

  virtual void WaitGPU() = 0;

  int InitializeLogicalDeviceAutoSelect(bool require_ray_tracing);

  int FramesInFlight() const;

  bool DebugEnabled() const;

  bool DeviceRayTracingSupport() const;

  std::string DeviceName() const;

 private:
  Settings settings_;

 protected:
  std::string device_name_{};
  bool ray_tracing_support_{false};
};

int CreateCore(BackendAPI api,
               const Core::Settings &settings,
               double_ptr<Core> pp_core);

}  // namespace grassland::graphics
