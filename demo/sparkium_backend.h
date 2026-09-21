#pragma once
#include <stdexcept>

#include "sparkium/backend/device.h"
#include "sparkium/renderer/renderer.h"

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

inline sparkium::RendererSettings SparkiumRendererSettings(sparkium::BackendSelection selection, bool debug = false) {
  sparkium::RendererSettings settings;
  settings.backend = selection.backend;
  settings.debug = debug;
  switch (selection.graphics_api) {
    case grassland::graphics::BACKEND_API_D3D12:
      settings.graphics_api = sparkium::GraphicsAPI::D3D12;
      break;
    case grassland::graphics::BACKEND_API_VULKAN:
      settings.graphics_api = sparkium::GraphicsAPI::Vulkan;
      break;
    case grassland::graphics::BACKEND_API_METAL:
      settings.graphics_api = sparkium::GraphicsAPI::Metal;
      break;
    default:
      break;
  }
  return settings;
}
