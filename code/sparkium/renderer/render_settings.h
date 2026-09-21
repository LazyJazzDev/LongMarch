#pragma once
#include <optional>

#include "sparkium/renderer/render_types.h"
#include "sparkium/scene/scene_definition.h"

namespace sparkium {
struct RendererSettings {
  RenderBackend backend{RenderBackend::Graphics};
  GraphicsAPI graphics_api{GraphicsAPI::Default};
  bool debug{};
};

struct RendererInfo {
  std::string device;
  bool ray_tracing{}, ray_query{};
};

struct RenderSettings {
  std::optional<RenderPipeline> pipeline;
  std::optional<int> samples_per_dispatch;
};

struct RenderImage {
  int width{}, height{}, accumulated_samples{};
  std::vector<uint8_t> rgba;
};

struct RenderProfile {
  std::map<std::string, double> cpu_ms, gpu_ms;
  std::map<std::string, uint64_t> counters;
};

}  // namespace sparkium
