#include "render_worker.h"

#include <chrono>

#include "../sparkium_backend.h"

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
  if (request.backend != request_.backend || request.graphics_api != request_.graphics_api)
    status_ = RenderStatus{};
  if (status_.name.empty()) {
    status_.backend = request.backend;
    status_.graphics_api = request.graphics_api;
  }
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
  std::unique_ptr<sparkium::Renderer> renderer;
  std::shared_ptr<const sparkium::SceneDefinition> scene;
  std::filesystem::path loaded_path;
  uint64_t loaded_revision = 0;
  std::optional<sparkium::RenderBackend> device_backend;
  auto device_graphics_api = graphics::BACKEND_API_DEFAULT;
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
        if (!device_backend || *device_backend != request.backend || device_graphics_api != request.graphics_api) {
          // Destroy every resource while its owning context is still current.
          // This runs only between dispatches, never on the window thread.
          renderer.reset();
          device_backend.reset();
          status = RenderStatus{};
          status.backend = request.backend;
          status.graphics_api = request.graphics_api;
        }
        status.revision = revision;
        status.error.clear();
        status.frame.reset();
        status.finished = false;
        status.updating = true;
        Publish(status);
        if (!scene || request.scene != loaded_path || request.reload != loaded_revision) {
          auto next = sparkium::LoadScene(request.scene);
          scene = std::move(next);
          loaded_path = request.scene;
          loaded_revision = request.reload;
        }
        if (Interrupted(revision))
          continue;
        if (!renderer) {
          renderer = sparkium::CreateRenderer(SparkiumRendererSettings({request.backend, request.graphics_api}));
          device_backend = request.backend;
          device_graphics_api = request.graphics_api;
        }
        if (renderer->GetScene() != scene)
          renderer->SetScene(scene);
        renderer->Configure({request.pipeline, request.samples});
        const auto info = renderer->Info();
        status.backend = request.backend;
        status.graphics_api = request.graphics_api;
        status.device = info.device;
        status.name = scene->name;
        status.pipeline = renderer->Pipeline();
        status.resolved_pipeline = renderer->ResolvePipeline(status.pipeline);
        status.automatic_pipeline = renderer->ResolvePipeline(sparkium::RENDER_PIPELINE_AUTO);
        status.ray_tracing = info.ray_tracing;
        status.ray_query = info.ray_query;
        status.samples = renderer->SamplesPerDispatch();
        Publish(status);
        waiting = false;
      }
      if (Interrupted(revision))
        continue;
      const auto start = std::chrono::steady_clock::now();
      renderer->Render();
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
        auto image = renderer->ReadImage();
        frame->width = image.width;
        frame->height = image.height;
        frame->accumulated_samples = image.accumulated_samples;
        frame->render_seconds = std::chrono::duration<double>(end - start).count();
        frame->rgba = std::move(image.rgba);
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
}

}  // namespace sparkium_gui
