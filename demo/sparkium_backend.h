#pragma once
#include <stdexcept>

#include "sparkium/backend/device.h"

inline sparkium::BackendSelection ParseSparkiumBackend(const std::string &name) {
  using namespace sparkium;
  BackendSelection selection;
  if (name == "cpu")
    selection.backend = RenderBackend::CPU;
  else if (name == "cuda")
    selection.backend = RenderBackend::CUDA;
  else if (name == "vulkan")
    selection.graphics_api = grassland::graphics::BACKEND_API_VULKAN;
  else if (name == "d3d12")
    selection.graphics_api = grassland::graphics::BACKEND_API_D3D12;
  else if (name == "metal")
    selection.graphics_api = grassland::graphics::BACKEND_API_METAL;
  else if (name != "auto" && name != "graphics")
    throw std::invalid_argument("unknown render backend: " + name);
  if (!SupportBackend(selection))
    throw std::runtime_error("render backend was not built: " + name);
  return selection;
}
