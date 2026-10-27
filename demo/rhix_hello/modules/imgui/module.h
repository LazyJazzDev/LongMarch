#pragma once
#include <chrono>

#include "../../module.h"
#include "grassland/util/fps_counter.h"

class ImGuiModule final : public ModuleBase {
 public:
  using ModuleBase::ModuleBase;

 private:
  int OnInit(const ModuleOptions &options) override;
  int OnUpdate() override;
  int OnRender() override;
  int OnTerminate() override;

  bool initialized_{};
  std::string capture_directory_{};
  grassland::rhix::Device device_{};
  grassland::rhix::DownloadToken last_frame_{};
  grassland::rhi::ImageDesc last_frame_desc_{};
  grassland::rhi::ColorSpace last_frame_space_{};
  float last_frame_white_scale_{};
  grassland::rhix::Queue queue_{};
  grassland::rhi::PresentMode requested_{};
  bool present_mode_changed_{};
  bool record_ui_{};
  grassland::rhix::Window window_{};
  int clicks_{};
  bool show_demo_{};
  grassland::FPSCounter fps_{};
  std::chrono::steady_clock::time_point title_time_{};
};
