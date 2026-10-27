#pragma once
#include <array>
#include <chrono>
#include <map>

#include "../../module.h"
#include "grassland/util/fps_counter.h"

class HDRModule final : public ModuleBase {
 public:
  using ModuleBase::ModuleBase;

 private:
  int OnInit(const ModuleOptions &options) override;
  int OnUpdate() override;
  int OnRender() override;
  int OnTerminate() override;

  bool initialized_{};
  int frames_{};
  std::string capture_directory_{};
  grassland::rhix::Device device_{};
  grassland::rhix::DownloadToken last_frame_{};
  grassland::rhi::ImageDesc last_frame_desc_{};
  grassland::rhi::ColorSpace last_frame_space_{};
  float last_frame_white_scale_{};
  grassland::rhix::Queue queue_{};
  size_t color_space_changes_{};
  bool present_mode_changed_{};
  grassland::rhi::PresentMode requested_{};
  grassland::rhix::Window window_{};
  std::vector<grassland::rhi::ColorSpace> spaces_{};
  size_t selected_{};
  std::string shader_{};
  std::map<std::pair<grassland::rhi::ImageFormat, grassland::rhi::ColorSpace>, grassland::rhix::Program> programs_{};
  grassland::rhi::BindingLayoutDesc settings_layout_{};
  grassland::rhix::Buffer settings_buffer_{};
  grassland::rhix::BindingGroup settings_group_{};
  float previous_scale_{};
  float previous_headroom_{};
  bool title_dirty_{};
  grassland::FPSCounter fps_{};
  std::chrono::steady_clock::time_point title_time_{};
  std::array<float, 4> settings_{};
  grassland::rhi::ColorSpace color_space_{};
};
