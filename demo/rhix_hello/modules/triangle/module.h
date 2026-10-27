#pragma once
#include <array>
#include <chrono>

#include "../../module.h"
#include "grassland/util/fps_counter.h"

class TriangleModule final : public ModuleBase {
 public:
  using ModuleBase::ModuleBase;

 private:
  int OnInit(const ModuleOptions &options) override;
  int OnUpdate() override;
  int OnRender() override;
  int OnTerminate() override;

  bool initialized_{};
  bool headless_{};
  std::string capture_directory_{};
  grassland::rhix::Device device_{};
  grassland::rhix::DownloadToken last_frame_{};
  grassland::rhi::ImageDesc last_frame_desc_{};
  grassland::rhi::ColorSpace last_frame_space_{};
  float last_frame_white_scale_{};
  grassland::rhi::PresentMode requested_{};
  bool present_mode_changed_{};
  grassland::rhix::Window window_{};
  grassland::rhix::Queue queue_{};
  std::array<grassland::rhix::Image, 3> targets_{};
  std::array<grassland::rhix::SubmissionToken, 3> pending_{};
  grassland::rhix::Program program_{};
  grassland::rhix::Buffer vertex_buffer_{};
  grassland::rhix::Buffer index_buffer_{};
  grassland::FPSCounter fps_{};
  std::chrono::steady_clock::time_point title_time_{};
  std::chrono::steady_clock::time_point benchmark_start_{};
};
