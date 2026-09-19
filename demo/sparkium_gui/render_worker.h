#pragma once

#include <long_march.h>

#include <condition_variable>
#include <mutex>
#include <optional>
#include <thread>

namespace sparkium_gui {

struct RenderRequest {
  grassland::graphics::BackendAPI backend = grassland::graphics::BACKEND_API_DEFAULT;
  std::filesystem::path scene;
  std::optional<sparkium::RenderPipeline> pipeline;
  std::optional<int> samples;
  uint64_t reload = 0;
};

// Only host pixels cross the boundary. No scene, device, or GPU resource is
// shared with the window thread; a published frame is immutable.
struct RenderFrame {
  uint64_t revision = 0;
  uint64_t number = 0;
  int width = 0, height = 0;
  int accumulated_samples = 0;
  double render_seconds = 0;
  std::vector<uint8_t> rgba;
};

struct RenderStatus {
  uint64_t revision = 0;
  bool updating = true;
  bool finished = false;
  bool ray_tracing = false, ray_query = false;
  grassland::graphics::BackendAPI backend = grassland::graphics::BACKEND_API_DEFAULT;
  sparkium::RenderPipeline pipeline = sparkium::RENDER_PIPELINE_AUTO;
  sparkium::RenderPipeline resolved_pipeline = sparkium::RENDER_PIPELINE_AUTO;
  sparkium::RenderPipeline automatic_pipeline = sparkium::RENDER_PIPELINE_AUTO;
  int samples = 1;
  std::string name, device, error;
  std::shared_ptr<const RenderFrame> frame;
};

class RenderWorker {
 public:
  explicit RenderWorker(int frame_limit = 0);
  ~RenderWorker();
  RenderWorker(const RenderWorker &) = delete;
  RenderWorker &operator=(const RenderWorker &) = delete;

  // Coalesces pending edits. Even a reset submits a new revision so an old
  // in-flight dispatch cannot publish pixels after the edit.
  uint64_t Submit(RenderRequest request);
  RenderStatus Snapshot() const;
  void Stop();

 private:
  void Run(int frame_limit);
  bool Publish(const RenderStatus &status);
  bool Interrupted(uint64_t revision) const;

  mutable std::mutex mutex_;
  std::condition_variable changed_;
  RenderRequest request_;
  RenderStatus status_;
  uint64_t revision_ = 0;
  bool stopping_ = false;
  std::thread thread_;
};

}  // namespace sparkium_gui
