#pragma once

#include <chrono>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>

#include "glm/gtc/constants.hpp"
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

 protected:
  float RotationAngle() {
    const auto now = std::chrono::steady_clock::now();
    if (!animation_start_)
      animation_start_ = now;
    const double seconds = std::chrono::duration<double>(now - *animation_start_).count();
    // Start at the first animation update, after loading. One revolution takes two seconds.
    return static_cast<float>(std::fmod(seconds, 2.0) * glm::pi<double>());
  }

 private:
  std::optional<std::chrono::steady_clock::time_point> animation_start_;
};

std::string LoadShader(const std::string &path);

}  // namespace graphics_hello
