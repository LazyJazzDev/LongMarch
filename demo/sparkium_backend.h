#pragma once
#include <stdexcept>
#include <string>

#include "grassland/graphics/graphics_util.h"
#include "sparkium/backends/offline_backend.h"

inline grassland::graphics::BackendAPI ParseSparkiumBackend(const std::string &name) {
  using namespace grassland::graphics;
  BackendAPI api;
  if (name == "auto")
    api = BACKEND_API_DEFAULT;
  else if (name == "metal")
    api = BACKEND_API_METAL;
  else if (name == "vulkan")
    api = BACKEND_API_VULKAN;
  else if (name == "d3d12")
    api = BACKEND_API_D3D12;
  else
    throw std::invalid_argument("unknown graphics backend: " + name);
  if (!SupportBackendAPI(api))
    throw std::runtime_error("graphics backend was not built: " + name);
  return api;
}

// True when the argument of --backend names an offline renderer ("cpu" or
// "cuda") instead of a graphics API. The offline backends shade on the CPU or
// with CUDA kernels and only use the graphics device as the scene's resource
// store, so they are selected with sparkium_backends instead of a graphics
// backend. The names are recognised even when the backend was not built, so
// that the failure explains which configuration enables it.
inline bool IsSparkiumOfflineBackend(const std::string &name) {
  return sparkium::backends::IsOfflineBackendName(name);
}
