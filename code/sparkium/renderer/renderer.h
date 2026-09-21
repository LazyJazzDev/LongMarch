#pragma once
#include <thread>

#include "sparkium/renderer/render_settings.h"

namespace sparkium {
namespace backend {
class Backend;
}

// Long-lived coordinator. All calls use its creating thread. Scene snapshots may
// be shared by independent renderers; backend resources never cross renderers.
class Renderer {
 public:
  Renderer();
  explicit Renderer(const RendererSettings &settings);
  ~Renderer();
  Renderer(const Renderer &) = delete;
  Renderer &operator=(const Renderer &) = delete;
  void SetBackend(const RendererSettings &settings);
  void ReleaseBackend();
  bool HasBackend() const;
  void SetScene(std::shared_ptr<const SceneDefinition> scene);
  std::shared_ptr<const SceneDefinition> GetScene() const;
  RendererInfo Info() const;
  bool SupportsPipeline(RenderPipeline pipeline) const;
  RenderPipeline ResolvePipeline(RenderPipeline pipeline) const;
  void Configure(const RenderSettings &settings);
  RenderPipeline Pipeline() const;
  int SamplesPerDispatch() const;
  void Reset();
  void Render();
  RenderImage ReadImage();
  std::vector<glm::vec4> ReadLinearImage();
  void BeginProfile(bool gpu_timestamps);
  RenderProfile EndProfile();

 private:
  void CheckThread() const;
  backend::Backend &Execution() const;
  std::thread::id thread_;
  std::shared_ptr<const SceneDefinition> scene_;
  RenderSettings settings_;
  std::unique_ptr<backend::Backend> backend_;
  bool profiling_{};
};

std::unique_ptr<Renderer> CreateRenderer(const RendererSettings &settings = {});
bool SupportRenderer(const RendererSettings &settings);
}  // namespace sparkium
