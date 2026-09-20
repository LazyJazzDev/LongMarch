#include "render_worker.h"

#include <chrono>

namespace sparkium_gui {
using namespace grassland;

RenderWorker::RenderWorker(int frame_limit) {
  thread_ = std::thread([this, frame_limit] { Run(frame_limit); });
}

RenderWorker::~RenderWorker() {
  Stop();
}

void RenderWorker::Stop() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stopping_ = true;
  }
  changed_.notify_all();
  if (thread_.joinable())
    thread_.join();
}

uint64_t RenderWorker::Submit(RenderRequest request) {
  if (request.samples && (*request.samples < 1 || *request.samples > 256))
    throw std::invalid_argument("samples per dispatch must be in [1, 256]");
  std::lock_guard<std::mutex> lock(mutex_);
  if (request.backend != request_.backend)
    status_ = RenderStatus{};
  if (status_.name.empty())
    status_.backend = request.backend;
  request_ = std::move(request);
  status_.revision = ++revision_;
  status_.updating = true;
  status_.finished = false;
  status_.error.clear();
  status_.frame.reset();
  changed_.notify_one();
  return revision_;
}

RenderStatus RenderWorker::Snapshot() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return status_;
}

bool RenderWorker::Interrupted(uint64_t revision) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return stopping_ || revision != revision_;
}

bool RenderWorker::Publish(const RenderStatus &status) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (stopping_ || status.revision != revision_)
    return false;
  status_ = status;
  return true;
}

void RenderWorker::Run(int frame_limit) {
  // Construction and destruction stay on this thread, including CUDA's
  // thread-local current context. The frontend never waits on this device.
  std::unique_ptr<graphics::Core> device;
  std::unique_ptr<sparkium::Core> renderer;
  std::unique_ptr<sparkium::JsonScene> scene;
  std::unique_ptr<graphics::Image> developed;
  std::optional<graphics::BackendAPI> device_backend;
  RenderRequest active;
  RenderStatus status;
  uint64_t applied = 0, frame_number = 0;
  bool waiting = true;
  auto last_preview = std::chrono::steady_clock::time_point::min();
  for (;;) {
    RenderRequest request;
    uint64_t revision;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      if (waiting)
        changed_.wait(lock, [&] { return stopping_ || revision_ != applied; });
      if (stopping_)
        break;
      request = request_;
      revision = revision_;
    }
    try {
      if (revision != applied) {
        applied = revision;
        if (!device_backend || *device_backend != request.backend) {
          // Destroy every resource while its owning context is still current.
          // This runs only between dispatches, never on the window thread.
          if (device)
            device->WaitGPU();
          developed.reset();
          scene.reset();
          renderer.reset();
          device.reset();
          device_backend.reset();
          status = RenderStatus{};
          status.backend = request.backend;
        }
        status.revision = revision;
        status.error.clear();
        status.frame.reset();
        status.finished = false;
        status.updating = true;
        Publish(status);
        if (!device) {
          if (!graphics::SupportBackendAPI(request.backend))
            throw std::runtime_error("selected render backend was not built");
          std::unique_ptr<graphics::Core> next_device;
          if (graphics::CreateCore(request.backend, {}, &next_device) != 0 || !next_device ||
              next_device->InitializeLogicalDeviceAutoSelect(false) != 0) {
            throw std::runtime_error("failed to initialize render backend");
          }
          device = std::move(next_device);
          device_backend = request.backend;
        }
        if (!renderer)
          renderer = std::make_unique<sparkium::Core>(device.get());
        if (Interrupted(revision))
          continue;
        if (!scene || request.scene != active.scene || request.reload != active.reload) {
          std::string error;
          auto next = sparkium::JsonScene::Load(renderer.get(), request.scene, &error);
          if (!next)
            throw std::runtime_error(error);
          device->WaitGPU();
          developed.reset();
          scene = std::move(next);
        }
        auto *film = scene->GetFilm();
        auto pipeline = request.pipeline.value_or(scene->GetRenderPipeline());
        const bool native = device->API() == graphics::BACKEND_API_CPU || device->API() == graphics::BACKEND_API_CUDA;
        if (native &&
            (pipeline == sparkium::RENDER_PIPELINE_RASTERIZATION || pipeline == sparkium::RENDER_PIPELINE_RAY_QUERY)) {
          // A scene's preferred graphics pipeline is not a restriction on
          // viewing it with a compute-only backend. Explicit requests fail.
          if (request.pipeline)
            throw std::runtime_error("CPU/CUDA do not support rasterization or inline ray queries");
          pipeline = sparkium::RENDER_PIPELINE_RT_FALLBACK;
        }
        if (request.samples)
          scene->GetScene()->settings.samples_per_dispatch = *request.samples;
        if (!developed && device->CreateImage(film->GetWidth(), film->GetHeight(),
                                              graphics::IMAGE_FORMAT_R8G8B8A8_UNORM, &developed) != 0)
          throw std::runtime_error("failed to create render image");
        film->Reset();
        active = request;
        status.backend = device->API();
        status.device = device->DeviceName();
        status.name = scene->GetName();
        status.pipeline = pipeline;
        status.resolved_pipeline = renderer->ResolveRenderPipeline(pipeline);
        status.automatic_pipeline = renderer->ResolveRenderPipeline(sparkium::RENDER_PIPELINE_AUTO);
        status.ray_tracing = device->DeviceRayTracingSupport();
        status.ray_query = device->DeviceRayQuerySupport();
        status.samples = scene->GetScene()->settings.samples_per_dispatch;
        Publish(status);
        waiting = false;
      }
      if (Interrupted(revision))
        continue;
      const auto start = std::chrono::steady_clock::now();
      auto *film = scene->GetFilm();
      renderer->Render(scene->GetScene(), scene->GetCamera(), film, status.pipeline);
      device->WaitGPU();
      const auto end = std::chrono::steady_clock::now();
      ++frame_number;
      if (Interrupted(revision))
        continue;
      status.finished = frame_limit > 0 && frame_number >= static_cast<uint64_t>(frame_limit);
      // Limit preview transfer to 30 Hz; sampling is independent of presentation.
      if (!status.frame || status.finished || end - last_preview >= std::chrono::milliseconds(33)) {
        auto frame = std::make_shared<RenderFrame>();
        frame->revision = revision;
        frame->number = frame_number;
        frame->width = film->GetWidth();
        frame->height = film->GetHeight();
        frame->accumulated_samples = film->info.accumulated_samples;
        frame->render_seconds = std::chrono::duration<double>(end - start).count();
        frame->rgba.resize(static_cast<size_t>(frame->width) * frame->height * 4);
        film->Develop(developed.get());
        developed->DownloadData(frame->rgba.data());
        status.frame = std::move(frame);
        status.updating = false;
        Publish(status);
        last_preview = std::chrono::steady_clock::now();
      }
      waiting = status.finished;
    } catch (const std::exception &error) {
      status.revision = revision;
      status.error = error.what();
      status.updating = false;
      status.finished = true;
      Publish(status);
      waiting = true;
    }
  }
  if (device) {
    try {
      device->WaitGPU();
    } catch (...) { /* Reported by the dispatch path. */
    }
  }
}

}  // namespace sparkium_gui
