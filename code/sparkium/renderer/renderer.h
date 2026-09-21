#pragma once

#include <optional>

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

// Scene-level interface, identical for GUI, CLI and library clients.
// Calls on one renderer are serialized on its creating thread. Independent
// renderers may run on different threads and share an immutable scene snapshot.
class Renderer {
 public:
  virtual ~Renderer() = default;
  virtual RendererInfo Info() const = 0;
  virtual void SetScene(std::shared_ptr<const SceneDefinition> scene) = 0;
  virtual std::shared_ptr<const SceneDefinition> GetScene() const = 0;
  virtual RenderPipeline ResolvePipeline(RenderPipeline pipeline) const = 0;
  // Changing settings resets accumulation without rereading or rebuilding scene data.
  virtual void Configure(const RenderSettings &settings) = 0;
  virtual RenderPipeline Pipeline() const = 0;
  virtual int SamplesPerDispatch() const = 0;
  virtual void Reset() = 0;
  virtual void Render() = 0;
  virtual RenderImage ReadImage() = 0;
  virtual std::vector<glm::vec4> ReadLinearImage() = 0;
  // Profiling remains part of the renderer; clients do not access device handles.
  virtual void BeginProfile(bool gpu_timestamps) = 0;
  virtual RenderProfile EndProfile() = 0;
};

std::unique_ptr<Renderer> CreateRenderer(const RendererSettings &settings = {});
bool SupportRenderer(const RendererSettings &settings);

}  // namespace sparkium
