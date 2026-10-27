#pragma once
#include <array>

#include "../../module.h"

class ImGuiMultiWindowModule final : public ModuleBase {
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
  std::array<grassland::rhix::DownloadToken, 2> last_frames_{};
  std::array<grassland::rhi::ImageDesc, 2> last_frame_descs_{};
  std::array<grassland::rhi::ColorSpace, 2> last_frame_spaces_{};
  std::array<float, 2> last_frame_white_scales_{};
  grassland::rhix::Queue queue_{};
  std::array<bool, 2> visible_{};
  std::array<grassland::rhi::ClearValue, 2> colors_{};
  std::array<grassland::rhix::Window, 2> windows_{};
  std::array<grassland::rhix::SubmissionToken, 2> pending_{};
  int clicks_{};
  std::array<char, 128> text_{};
  float speed_{};
  float amplitude_{};
  float phase_{};
  bool paused_{};
};
