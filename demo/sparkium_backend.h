#pragma once
#include <stdexcept>

#include "grassland/graphics/graphics_util.h"
#include "sparkium/core/core_util.h"

// `--backend` names a graphics API for the GPU pipelines, plus the two native
// renderers `cpu` and `cuda`, which do their rendering computation on the host
// or in CUDA kernels. A native backend still needs a graphics core to read
// geometry and textures back, so it keeps the default graphics API.
struct SparkiumBackend {
  grassland::graphics::BackendAPI api{grassland::graphics::BACKEND_API_DEFAULT};
  bool native{false};
  sparkium::RenderPipeline pipeline{sparkium::RENDER_PIPELINE_AUTO};
};

inline SparkiumBackend ParseSparkiumBackendSelection(const std::string &name) {
  using namespace grassland::graphics;
  SparkiumBackend backend;
  if (name == "cpu") {
    backend.native = true;
    backend.pipeline = sparkium::RENDER_PIPELINE_NATIVE_CPU;
    return backend;
  }
  if (name == "cuda") {
    backend.native = true;
    backend.pipeline = sparkium::RENDER_PIPELINE_NATIVE_CUDA;
    return backend;
  }
  if (name == "auto")
    backend.api = BACKEND_API_DEFAULT;
  else if (name == "metal")
    backend.api = BACKEND_API_METAL;
  else if (name == "vulkan")
    backend.api = BACKEND_API_VULKAN;
  else if (name == "d3d12")
    backend.api = BACKEND_API_D3D12;
  else
    throw std::invalid_argument("unknown graphics backend: " + name);
  if (!SupportBackendAPI(backend.api))
    throw std::runtime_error("graphics backend was not built: " + name);
  return backend;
}

// Graphics-API-only entry point, for callers that cannot host a native
// renderer.
inline grassland::graphics::BackendAPI ParseSparkiumBackend(const std::string &name) {
  auto backend = ParseSparkiumBackendSelection(name);
  if (backend.native)
    throw std::invalid_argument("not a graphics backend: " + name);
  return backend.api;
}
