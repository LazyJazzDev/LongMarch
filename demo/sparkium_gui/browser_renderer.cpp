#include "browser_renderer.h"

#include <chrono>
#include <deque>

using namespace grassland::rhi;

BrowserRenderer::BrowserRenderer(IDevice *device)
    : device_(device),
      thread_([this] {
        try {
          Run();
        } catch (const std::exception &error) {
          std::lock_guard lock(mutex_);
          state_.error = error.what();
          state_.revision = requested_.revision;
        }
      }) {
}

BrowserRenderer::~BrowserRenderer() {
  stopping_ = true;
  condition_.notify_all();
  thread_.join();
}

uint64_t BrowserRenderer::Load(const std::filesystem::path &path, bool hdr) {
  std::lock_guard lock(mutex_);
  ++requested_.revision;
  ++requested_.scene_revision;
  ++requested_.reset_revision;
  requested_.path = path;
  requested_.settings.reset();
  requested_.hdr = hdr;
  condition_.notify_all();
  return requested_.revision;
}

uint64_t BrowserRenderer::Configure(const Settings &settings, bool reset) {
  std::lock_guard lock(mutex_);
  ++requested_.revision;
  requested_.reset_revision += reset;
  requested_.settings = settings;
  requested_.hdr = settings.hdr;
  condition_.notify_all();
  return requested_.revision;
}

BrowserRenderer::State BrowserRenderer::GetState() const {
  std::lock_guard lock(mutex_);
  return state_;
}

std::shared_ptr<const BrowserRenderer::Frame> BrowserRenderer::Latest() const {
  std::lock_guard lock(mutex_);
  return latest_;
}

void BrowserRenderer::Presented(const std::shared_ptr<const Frame> &frame, ObjPtr<ISubmissionToken> completion) {
  std::lock_guard lock(mutex_);
  for (auto &slot : slots_)
    if (slot.frame == frame) {
      slot.displayed = std::move(completion);
      condition_.notify_all();
      break;
    }
}

void BrowserRenderer::Fail(uint64_t revision, const std::string &error) {
  std::lock_guard lock(mutex_);
  if (requested_.revision == revision) {
    state_.revision = revision;
    state_.error = error;
  }
}

void BrowserRenderer::Run() {
  std::unique_ptr<sparkium::Core> core;
  std::unique_ptr<sparkium::JsonScene> scene;
  std::unique_ptr<sparkium::Renderer> renderer;
  ObjPtr<IQueue> queue = device_->GetQueue(QUEUE_TYPE_ASSIST);
  Request active;
  Settings settings;
  uint64_t sequence{};
  std::deque<std::shared_ptr<Frame>> pending;
  auto retire = [&]() {
    auto frame = std::move(pending.front());
    pending.pop_front();
    const bool completed = frame->rendered->WaitHost();
    std::lock_guard lock(mutex_);
    if (!completed) {
      state_.error = "GPU render submission failed";
      return false;
    }
    if (frame->revision == requested_.revision && (!latest_ || frame->sequence > latest_->sequence)) {
      latest_ = frame;
      state_.completed_samples = frame->samples;
    }
    return true;
  };
  while (!stopping_.load()) {
    Request request;
    {
      std::unique_lock lock(mutex_);
      if (!requested_.revision) {
        condition_.wait(lock, [&] { return stopping_.load() || requested_.revision; });
        continue;
      }
      request = requested_;
    }
    if (request.revision != active.revision) {
      // Drain only this worker's bounded tasks before replacing CPU scene state.
      while (!pending.empty())
        if (!retire())
          return;
      if (request.scene_revision != active.scene_revision) {
        renderer.reset();
        scene.reset();
        core = sparkium::Core::Create(device_);
        std::string error;
        if (core)
          scene = sparkium::JsonScene::Load(core.get(), request.path, &error);
        if (!scene) {
          Fail(request.revision, error.empty() ? "Cannot create scene resources" : error);
          active = request;
          continue;
        }
        renderer = std::make_unique<sparkium::Renderer>(core.get());
        settings.ray = scene->GetScene()->settings.raytracing;
        settings.ray.samples_per_dispatch = std::clamp(settings.ray.samples_per_dispatch, 1, 4);
        settings.raster = scene->GetScene()->settings.raster;
        settings.film = scene->GetFilm()->info;
        settings.pipeline = scene->GetRenderPipeline();
      }
      if (request.settings)
        settings = *request.settings;
      settings.hdr = request.hdr;
      if (scene) {
        auto *film = scene->GetFilm();
        const int samples = film->info.accumulated_samples;
        film->info = settings.film;
        film->info.accumulated_samples = samples;
        scene->GetScene()->settings.raytracing = settings.ray;
        scene->GetScene()->settings.raster = settings.raster;
        if (request.reset_revision != active.reset_revision)
          film->Reset();
        std::lock_guard lock(mutex_);
        if (requested_.revision == request.revision)
          state_ = {request.revision,
                    scene->GetName(),
                    {},
                    static_cast<uint32_t>(film->GetWidth()),
                    static_cast<uint32_t>(film->GetHeight()),
                    film->info.accumulated_samples,
                    settings};
      }
      active = request;
    }
    if (!scene) {
      std::unique_lock lock(mutex_);
      condition_.wait(lock, [&] { return stopping_.load() || requested_.revision != active.revision; });
      continue;
    }
    if (pending.size() == 2) {
      if (!retire())
        return;
      continue;
    }
    std::shared_ptr<Frame> frame;
    ObjPtr<ISubmissionToken> displayed;
    {
      std::lock_guard lock(mutex_);
      for (auto &slot : slots_) {
        // A UI lease or published latest image cannot be overwritten. The last
        // Main submission must also have finished before this slot is recycled.
        if (slot.frame && slot.frame.use_count() != 1)
          continue;
        if (slot.displayed && slot.displayed->GetStatus() == SUBMISSION_STATUS_PENDING)
          continue;
        if (slot.displayed && slot.displayed->GetStatus() != SUBMISSION_STATUS_COMPLETED) {
          state_.error = "Presentation submission failed";
          return;
        }
        if (!slot.frame)
          slot.frame = std::make_shared<Frame>();
        frame = slot.frame;
        displayed = slot.displayed;
        break;
      }
    }
    if (!frame) {
      if (!pending.empty()) {
        if (!retire())
          return;
      } else {
        std::unique_lock lock(mutex_);
        condition_.wait_for(lock, std::chrono::milliseconds(1));
      }
      continue;
    }
    const auto width = static_cast<uint32_t>(scene->GetFilm()->GetWidth());
    const auto height = static_cast<uint32_t>(scene->GetFilm()->GetHeight());
    const auto format = settings.hdr ? IMAGE_FORMAT_R32G32B32A32_SFLOAT : IMAGE_FORMAT_R8G8B8A8_UNORM;
    if (!frame->image || frame->image->GetDesc().width != width || frame->image->GetDesc().height != height ||
        frame->image->GetDesc().format != format) {
      frame->image = device_->CreateImage({width, height, format});
      ObjPtr<ICommandContext> initialize = queue ? queue->CreateCommandContext() : nullptr;
      if (!frame->image || !initialize || !initialize->CmdClearImage(frame->image.Get(), {})) {
        Fail(active.revision, "Cannot initialize display image");
        return;
      }
      ObjPtr<ISubmissionToken> initialized = initialize->Submit();
      if (!initialized) {
        Fail(active.revision, "Cannot submit display initialization");
        return;
      }
    }
    SubmitInfo dependencies;
    if (displayed)
      dependencies.waits.push_back(displayed->GetEvent());
    // Keep individual dispatches short; sample indices and accumulation still
    // advance per sample, using the same shader math and random sequence.
    scene->GetScene()->settings.samples_per_dispatch = std::min(settings.ray.samples_per_dispatch, 4);
    frame->revision = active.revision;
    frame->sequence = ++sequence;
    frame->hdr = settings.hdr;
    frame->rendered = renderer->Submit(scene->GetScene(), scene->GetCamera(), scene->GetFilm(), settings.pipeline,
                                       frame->image.Get(), settings.hdr, dependencies);
    if (!frame->rendered) {
      Fail(active.revision, "Cannot submit rendering");
      return;
    }
    frame->samples = scene->GetFilm()->info.accumulated_samples;
    pending.push_back(std::move(frame));
  }
  while (!pending.empty())
    if (!retire())
      return;
}
