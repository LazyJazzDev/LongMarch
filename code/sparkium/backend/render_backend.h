#pragma once
#include "sparkium/renderer/render_settings.h"

namespace sparkium::backend {
// Owns execution objects translated from a scene, never file loading or frontend state.
class Backend {
 public:
  virtual ~Backend() = default;
  virtual RendererInfo Info() const = 0;
  virtual void SetScene(std::shared_ptr<const SceneDefinition> scene) = 0;
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

  virtual bool SupportsPipeline(RenderPipeline pipeline) const = 0;
};

std::unique_ptr<Backend> CreateBackend(const RendererSettings &settings);
}  // namespace sparkium::backend
