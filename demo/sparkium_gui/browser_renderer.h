#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <thread>

#include "sparkium/sparkium.h"

// Scene browser state crosses threads only as values or completed image leases.
class BrowserRenderer {
 public:
  struct Settings {
    sparkium::Scene::Settings::RayTracing ray;
    sparkium::Scene::Settings::Rasterization raster;
    sparkium::Film::Info film;
    sparkium::RenderPipeline pipeline{sparkium::RENDER_PIPELINE_AUTO};
    bool hdr{};
  };

  struct State {
    uint64_t revision{};
    std::string name, error;
    uint32_t width{}, height{};
    int completed_samples{};
    Settings settings;
  };

  struct Frame {
    grassland::rhi::ObjPtr<grassland::rhi::IImage> image;
    grassland::rhi::ObjPtr<grassland::rhi::ISubmissionToken> rendered;
    uint64_t revision{}, sequence{};
    int samples{};
    bool hdr{};
  };

  explicit BrowserRenderer(grassland::rhi::IDevice *device);
  ~BrowserRenderer();
  uint64_t Load(const std::filesystem::path &path, bool hdr);
  uint64_t Configure(const Settings &settings, bool reset);
  State GetState() const;
  std::shared_ptr<const Frame> Latest() const;
  void Presented(const std::shared_ptr<const Frame> &frame,
                 grassland::rhi::ObjPtr<grassland::rhi::ISubmissionToken> completion);

 private:
  struct Request {
    uint64_t revision{}, scene_revision{}, reset_revision{};
    std::filesystem::path path;
    std::optional<Settings> settings;
    bool hdr{};
  };

  struct Slot {
    std::shared_ptr<Frame> frame;
    grassland::rhi::ObjPtr<grassland::rhi::ISubmissionToken> displayed;
  };

  void Run();
  void Fail(uint64_t revision, const std::string &error);

  grassland::rhi::IDevice *device_;
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  Request requested_;
  State state_;
  std::array<Slot, 3> slots_;
  std::shared_ptr<Frame> latest_;
  std::atomic<bool> stopping_{};
  std::thread thread_;
};
