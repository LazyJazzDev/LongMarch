#pragma once

#include <stdexcept>
#include <string>

#include "long_march.h"

namespace graphics_hello {

inline std::string GraphicsHelloTitle(grassland::graphics::BackendAPI api) {
  return std::string("[") + grassland::graphics::BackendAPIString(api) + "]";
}

inline void InitializeGraphicsHello(grassland::graphics::BackendAPI api,
                                    std::shared_ptr<grassland::graphics::Core> &core,
                                    bool require_ray_tracing = false) {
  using namespace grassland::graphics;
  if (!SupportBackendAPI(api) || CreateCore(api, Core::Settings{}, &core) != 0 || !core)
    throw std::runtime_error("Requested graphics backend is unavailable");
  if (core->InitializeLogicalDeviceAutoSelect(require_ray_tracing) != 0)
    throw std::runtime_error("No compatible graphics device found");
  grassland::LogInfo("Backend API: {}", BackendAPIString(core->API()));
  grassland::LogInfo("Device Name: {}", core->DeviceName());
  grassland::LogInfo("- Ray Tracing Support: {}", core->DeviceRayTracingSupport());
  grassland::LogInfo("- Ray Query Support: {}", core->DeviceRayQuerySupport());
}

class Module {
 public:
  virtual ~Module() = default;
  virtual void OnInit() = 0;
  virtual void OnClose() = 0;
  virtual void OnUpdate() = 0;
  virtual void OnRender() = 0;
  virtual grassland::graphics::Window *GetWindow() const = 0;
  virtual bool IsAlive() const = 0;
};

std::string LoadShader(const std::string &path);

}  // namespace graphics_hello
