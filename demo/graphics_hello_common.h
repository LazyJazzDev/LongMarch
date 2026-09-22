#pragma once

#include <iostream>
#include <stdexcept>
#include <string>

#include "long_march.h"

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

// A frame limit makes it possible to run the same demo in automated GPU smoke tests.
template <typename Application>
int RunGraphicsHello(int argc, char **argv) {
  using namespace grassland::graphics;
  try {
    BackendAPI api = BACKEND_API_DEFAULT;
    int frames = 0;
    for (int i = 1; i < argc; ++i) {
      std::string option = argv[i];
      if (option == "--help") {
        std::cout << "Usage: " << argv[0] << " [--backend auto|metal|vulkan|d3d12] [--frames N]\n";
        return 0;
      }
      if (option == "--backend" && i + 1 < argc) {
        std::string name = argv[++i];
        if (name == "auto")
          api = BACKEND_API_DEFAULT;
        else if (name == "metal")
          api = BACKEND_API_METAL;
        else if (name == "vulkan")
          api = BACKEND_API_VULKAN;
        else if (name == "d3d12")
          api = BACKEND_API_D3D12;
        else
          throw std::invalid_argument("Unknown backend: " + name);
      } else if (option == "--frames" && i + 1 < argc) {
        std::string value = argv[++i];
        size_t parsed = 0;
        frames = std::stoi(value, &parsed);
        if (parsed != value.size() || frames <= 0)
          throw std::invalid_argument("--frames requires a positive integer");
      } else {
        throw std::invalid_argument("Unknown or incomplete option: " + option);
      }
    }
    Application app{api};
    app.OnInit();
    int rendered = 0;
    while (app.IsAlive() && (!frames || rendered < frames)) {
      glfwPollEvents();
      app.OnUpdate();
      if (app.IsAlive()) {
        app.OnRender();
        ++rendered;
      }
    }
    app.OnClose();
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
