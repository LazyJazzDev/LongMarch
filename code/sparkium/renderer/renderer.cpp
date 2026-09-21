#include "sparkium/renderer/renderer.h"

#include "sparkium/backend/device.h"
#include "sparkium/backend/render_backend.h"

namespace sparkium {
Renderer::Renderer() : thread_(std::this_thread::get_id()) {
}

Renderer::Renderer(const RendererSettings &settings) : Renderer() {
  SetBackend(settings);
}

Renderer::~Renderer() = default;

void Renderer::CheckThread() const {
  if (std::this_thread::get_id() != thread_)
    throw std::logic_error("renderer calls must use its creating thread");
}

backend::Backend &Renderer::Execution() const {
  CheckThread();
  if (!backend_)
    throw std::logic_error("renderer has no backend");
  return *backend_;
}

void Renderer::SetBackend(const RendererSettings &settings) {
  CheckThread();
  if (profiling_)
    throw std::logic_error("cannot switch backend during profiling");
  // Release resources before initializing another CUDA/graphics context.
  // The scene and requested settings survive even if backend initialization fails.
  backend_.reset();
  auto next = backend::CreateBackend(settings);
  if (settings_.pipeline)
    next->ResolvePipeline(*settings_.pipeline);
  if (scene_) {
    next->SetScene(scene_);
    next->Configure(settings_);
  }
  backend_ = std::move(next);
}

void Renderer::ReleaseBackend() {
  CheckThread();
  if (profiling_)
    throw std::logic_error("cannot release backend during profiling");
  backend_.reset();
}

bool Renderer::HasBackend() const {
  CheckThread();
  return bool(backend_);
}

void Renderer::SetScene(std::shared_ptr<const SceneDefinition> scene) {
  CheckThread();
  if (profiling_)
    throw std::logic_error("cannot change scene during profiling");
  if (!scene)
    throw std::invalid_argument("null scene definition");
  scene->Validate();
  if (backend_) {
    backend_->SetScene(scene);
    backend_->Configure(settings_);
  }
  scene_ = std::move(scene);
}

std::shared_ptr<const SceneDefinition> Renderer::GetScene() const {
  CheckThread();
  return scene_;
}

RendererInfo Renderer::Info() const {
  return Execution().Info();
}

bool Renderer::SupportsPipeline(RenderPipeline pipeline) const {
  return Execution().SupportsPipeline(pipeline);
}

RenderPipeline Renderer::ResolvePipeline(RenderPipeline pipeline) const {
  return Execution().ResolvePipeline(pipeline);
}

void Renderer::Configure(const RenderSettings &settings) {
  CheckThread();
  if (settings.samples_per_dispatch && *settings.samples_per_dispatch <= 0)
    throw std::invalid_argument("samples per dispatch must be positive");
  if (settings.pipeline &&
      (*settings.pipeline < RENDER_PIPELINE_RASTERIZATION || *settings.pipeline > RENDER_PIPELINE_RAY_QUERY))
    throw std::invalid_argument("invalid render pipeline");
  if (backend_) {
    if (settings.pipeline)
      backend_->ResolvePipeline(*settings.pipeline);
    if (scene_)
      backend_->Configure(settings);
  }
  settings_ = settings;
}

RenderPipeline Renderer::Pipeline() const {
  return Execution().Pipeline();
}

int Renderer::SamplesPerDispatch() const {
  return Execution().SamplesPerDispatch();
}

void Renderer::Reset() {
  Execution().Reset();
}

void Renderer::Render() {
  Execution().Render();
}

RenderImage Renderer::ReadImage() {
  return Execution().ReadImage();
}

std::vector<glm::vec4> Renderer::ReadLinearImage() {
  return Execution().ReadLinearImage();
}

void Renderer::BeginProfile(bool gpu_timestamps) {
  Execution().BeginProfile(gpu_timestamps);
  profiling_ = true;
}

RenderProfile Renderer::EndProfile() {
  auto result = Execution().EndProfile();
  profiling_ = false;
  return result;
}

std::unique_ptr<Renderer> CreateRenderer(const RendererSettings &settings) {
  return std::make_unique<Renderer>(settings);
}

bool SupportRenderer(const RendererSettings &settings) {
  grassland::graphics::BackendAPI api;
  switch (settings.graphics_api) {
    case GraphicsAPI::Default:
      api = grassland::graphics::BACKEND_API_DEFAULT;
      break;
    case GraphicsAPI::D3D12:
      api = grassland::graphics::BACKEND_API_D3D12;
      break;
    case GraphicsAPI::Vulkan:
      api = grassland::graphics::BACKEND_API_VULKAN;
      break;
    case GraphicsAPI::Metal:
      api = grassland::graphics::BACKEND_API_METAL;
      break;
    default:
      return false;
  }
  return SupportBackend({settings.backend, api});
}
}  // namespace sparkium
