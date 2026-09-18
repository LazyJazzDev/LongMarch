#pragma once
#include <stdexcept>

#include "grassland/graphics/graphics_util.h"

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
  else if (name == "host")
    api = BACKEND_API_HOST;
  else
    throw std::invalid_argument("unknown graphics backend: " + name);
  if (!SupportBackendAPI(api))
    throw std::runtime_error("graphics backend was not built: " + name);
  return api;
}
