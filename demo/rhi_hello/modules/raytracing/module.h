#pragma once
#include <array>
#include <chrono>

#include "../../module.h"
#include "grassland/util/fps_counter.h"

class RayTracingModule final : public ModuleBase {
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
  grassland::rhi::IDevice *device_{};
  grassland::rhi::ObjPtr<grassland::rhi::IDownloadToken> last_frame_{};
  grassland::rhi::ImageDesc last_frame_desc_{};
  grassland::rhi::ColorSpace last_frame_space_{};
  float last_frame_white_scale_{};
  grassland::rhi::PresentMode requested_{};
  bool present_mode_changed_{};
  grassland::rhi::ObjPtr<grassland::rhi::IWindow> window_{};
  grassland::rhi::ObjPtr<grassland::rhi::IQueue> queue_{};
  std::array<grassland::rhi::ObjPtr<grassland::rhi::IImage>, 3> targets_{};
  std::array<grassland::rhi::ObjPtr<grassland::rhi::ISubmissionToken>, 3> pending_{};
  grassland::rhi::ObjPtr<grassland::rhi::IRayTracingProgram> trace_{};
  grassland::rhi::ObjPtr<grassland::rhi::IProgram> program_{};
  grassland::rhi::ObjPtr<grassland::rhi::ITopLevelAccelerationStructure> tlas_{};
  std::vector<grassland::rhi::RayTracingInstance> instances_{};
  grassland::rhi::ObjPtr<grassland::rhi::IBuffer> camera_buffer_{};
  grassland::rhi::ObjPtr<grassland::rhi::IBindingGroup> trace_group_{};
  grassland::rhi::ObjPtr<grassland::rhi::IBindingGroup> display_group_{};
  grassland::rhi::ObjPtr<grassland::rhi::IImage> output_{};
  grassland::FPSCounter fps_{};
  std::chrono::steady_clock::time_point title_time_{};
  std::chrono::steady_clock::time_point benchmark_start_{};
  std::chrono::steady_clock::time_point animation_start_{};
};
