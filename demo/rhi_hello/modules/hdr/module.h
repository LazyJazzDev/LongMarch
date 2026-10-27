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
  grassland::rhi::IDevice *device_{};
  grassland::rhi::ObjPtr<grassland::rhi::IDownloadToken> last_frame_{};
  grassland::rhi::ImageDesc last_frame_desc_{};
  grassland::rhi::ColorSpace last_frame_space_{};
  float last_frame_white_scale_{};
  grassland::rhi::ObjPtr<grassland::rhi::IQueue> queue_{};
  size_t color_space_changes_{};
  bool present_mode_changed_{};
  grassland::rhi::PresentMode requested_{};
  grassland::rhi::ObjPtr<grassland::rhi::IWindow> window_{};
  std::vector<grassland::rhi::ColorSpace> spaces_{};
  size_t selected_{};
  std::string shader_{};
  std::map<std::pair<grassland::rhi::ImageFormat, grassland::rhi::ColorSpace>,
           grassland::rhi::ObjPtr<grassland::rhi::IProgram>>
      programs_{};
  grassland::rhi::BindingLayoutDesc settings_layout_{};
  grassland::rhi::ObjPtr<grassland::rhi::IBuffer> settings_buffer_{};
  grassland::rhi::ObjPtr<grassland::rhi::IBindingGroup> settings_group_{};
  float previous_scale_{};
  float previous_headroom_{};
  bool title_dirty_{};
  grassland::FPSCounter fps_{};
  std::chrono::steady_clock::time_point title_time_{};
  std::array<float, 4> settings_{};
  grassland::rhi::ColorSpace color_space_{};
};
