#pragma once
#include <stdexcept>

#include "grassland/rhi/rhi_util.h"

inline grassland::rhi::BackendAPI ParseSparkiumBackend(const std::string &name) {
  using namespace grassland::rhi;
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
  return api;
}
