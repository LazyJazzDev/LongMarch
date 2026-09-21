#pragma once
#include <chrono>
#include <thread>

#include "grassland/graphics/frame_profile.h"
#include "sparkium/backend/common/scene_objects.h"
#include "sparkium/backend/render_backend.h"

namespace sparkium::backend {
inline graphics::BackendAPI GraphicsBackend(GraphicsAPI api) {
  switch (api) {
    case GraphicsAPI::Default:
      return graphics::BACKEND_API_DEFAULT;
    case GraphicsAPI::D3D12:
      return graphics::BACKEND_API_D3D12;
    case GraphicsAPI::Vulkan:
      return graphics::BACKEND_API_VULKAN;
    case GraphicsAPI::Metal:
      return graphics::BACKEND_API_METAL;
  }
  throw std::invalid_argument("invalid graphics API");
}

class ExecutionBackend : public Backend {
 public:
  explicit ExecutionBackend(const RendererSettings &settings) : thread_(std::this_thread::get_id()) {
    if (CreateDevice({settings.backend, GraphicsBackend(settings.graphics_api)}, {2, settings.debug}, &device_) ||
        device_->InitializeLogicalDeviceAutoSelect(false))
      throw std::runtime_error("failed to initialize rendering backend");
    core_ = std::make_unique<Core>(device_.get());
  }

  ~ExecutionBackend() override {
    try {
      device_->WaitGPU();
    } catch (...) {
    }
  }

  RendererInfo Info() const override {
    CheckThread();
    return {device_->DeviceName(), device_->DeviceRayTracingSupport(), device_->DeviceRayQuerySupport()};
  }

  void SetScene(std::shared_ptr<const SceneDefinition> definition) override {
    CheckThread();
    if (!definition)
      throw std::invalid_argument("null scene definition");
    if (instance_ && instance_->definition == definition) {
      Reset();
      return;
    }
    auto next = CreateSceneObjects(core_.get(), std::move(definition));
    device_->WaitGPU();
    developed_.reset();
    instance_ = std::move(next);
    Configure({});
  }

  RenderPipeline ResolvePipeline(RenderPipeline pipeline) const override {
    CheckThread();
    if (pipeline < RENDER_PIPELINE_RASTERIZATION || pipeline > RENDER_PIPELINE_RAY_QUERY)
      throw std::invalid_argument("invalid render pipeline");
    if (!SupportsPipeline(pipeline))
      throw std::invalid_argument("selected backend does not support this pipeline");
    return pipeline == RENDER_PIPELINE_AUTO ? DefaultPipeline() : pipeline;
  }

  void Configure(const RenderSettings &settings) override {
    CheckScene();
    auto pipeline = settings.pipeline.value_or(RENDER_PIPELINE_AUTO);
    ResolvePipeline(pipeline);
    const int samples = settings.samples_per_dispatch.value_or(instance_->definition->integrator.samples_per_dispatch);
    if (samples <= 0)
      throw std::invalid_argument("samples per dispatch must be positive");
    pipeline_ = pipeline;
    instance_->scene->settings.samples_per_dispatch = samples;
    Reset();
  }

  RenderPipeline Pipeline() const override {
    CheckScene();
    return pipeline_;
  }

  int SamplesPerDispatch() const override {
    CheckScene();
    return instance_->scene->settings.samples_per_dispatch;
  }

  void Reset() override {
    CheckScene();
    device_->WaitGPU();
    instance_->film->Reset();
    instance_->film->info.accumulated_samples = 0;
  }

  void Render() override {
    CheckScene();
    graphics::CpuProfileScope scope("render_wall");
    Dispatch(core_.get(), instance_.get(), ResolvePipeline(pipeline_));
    device_->WaitGPU();
  }

  RenderImage ReadImage() override {
    CheckScene();
    auto *film = instance_->film.get();
    if (!developed_ &&
        device_->CreateImage(film->GetWidth(), film->GetHeight(), graphics::IMAGE_FORMAT_R8G8B8A8_UNORM, &developed_))
      throw std::runtime_error("failed to create developed image");
    film->Develop(developed_.get());
    RenderImage result{film->GetWidth(), film->GetHeight(), film->info.accumulated_samples, {}};
    result.rgba.resize(size_t(result.width) * result.height * 4);
    developed_->DownloadData(result.rgba.data());
    return result;
  }

  std::vector<glm::vec4> ReadLinearImage() override {
    CheckScene();
    auto *film = instance_->film.get();
    std::vector<glm::vec4> result(size_t(film->GetWidth()) * film->GetHeight());
    device_->WaitGPU();
    film->GetRawImage()->DownloadData(result.data());
    return result;
  }

  void BeginProfile(bool gpu_timestamps) override {
    CheckThread();
    if (profiling_)
      throw std::logic_error("profile already active");
    if (gpu_timestamps && !gpu_profile_) {
      if (!device_->GraphicsCore())
        throw std::invalid_argument("CPU/CUDA profiling requires CPU timings");
      profile_ = std::make_unique<graphics::FrameProfile>(device_->GraphicsCore());
      gpu_profile_ = true;
    } else if (!profile_) {
      profile_ = std::make_unique<graphics::FrameProfile>(device_->DeviceName());
    }
    profile_->Begin(gpu_timestamps);
    profile_start_ = std::chrono::steady_clock::now();
    profiling_ = true;
  }

  RenderProfile EndProfile() override {
    CheckThread();
    if (!profiling_)
      throw std::logic_error("profile is not active");
    device_->WaitGPU();
    const auto elapsed = std::chrono::steady_clock::now() - profile_start_;
    profile_->Finish();
    RenderProfile result;
    result.cpu_ms.insert(profile_->cpu_ms.begin(), profile_->cpu_ms.end());
    result.gpu_ms.insert(profile_->gpu_ms.begin(), profile_->gpu_ms.end());
    result.counters.insert(profile_->counters.begin(), profile_->counters.end());
    result.cpu_ms["frame_wall"] = std::chrono::duration<double, std::milli>(elapsed).count();
    profiling_ = false;
    return result;
  }

 protected:
  virtual RenderPipeline DefaultPipeline() const = 0;

  backend::Device *BackendDevice() const {
    return device_.get();
  }

  virtual std::unique_ptr<SceneObjects> CreateSceneObjects(Core *core,
                                                           std::shared_ptr<const SceneDefinition> scene) = 0;
  virtual void Dispatch(Core *core, SceneObjects *scene, RenderPipeline pipeline) = 0;

 private:
  void CheckThread() const {
    if (std::this_thread::get_id() != thread_)
      throw std::logic_error("renderer calls must use its creating thread");
  }

  void CheckScene() const {
    CheckThread();
    if (!instance_)
      throw std::logic_error("renderer has no scene");
  }

  std::thread::id thread_;
  std::unique_ptr<backend::Device> device_;
  std::unique_ptr<Core> core_;
  std::unique_ptr<SceneObjects> instance_;
  std::unique_ptr<graphics::Image> developed_;
  std::unique_ptr<graphics::FrameProfile> profile_;
  bool profiling_{}, gpu_profile_{};
  std::chrono::steady_clock::time_point profile_start_;
  RenderPipeline pipeline_{RENDER_PIPELINE_AUTO};
};

}  // namespace sparkium::backend
