#include "module.h"

#include <utility>

ModuleBase::ModuleBase(grassland::rhix::Core core, grassland::rhix::Device device)
    : core_(std::move(core)),
      device_(std::move(device)) {
}

int ModuleBase::Run(const ModuleOptions &options) {
  if (!core_ || !device_)
    return 1;
  should_close_ = false;
  frame_index_ = 0;
  int result = OnInit(options);
  while (!result && !should_close_ && (!options.frames || frame_index_ < options.frames)) {
    if (!options.headless && !grassland::rhi::PollEvents()) {
      RequestClose();
      break;
    }
    result = OnUpdate();
    if (!result && !should_close_)
      result = OnRender();
  }
  RequestClose();
  const int termination_result = OnTerminate();
  return result ? result : termination_result;
}

const grassland::rhix::Core &ModuleBase::GetCore() const {
  return core_;
}

const grassland::rhix::Device &ModuleBase::GetDevice() const {
  return device_;
}

grassland::rhi::BackendAPI ModuleBase::GetBackendAPI() const {
  return core_.GetBackendAPI();
}

bool ModuleBase::ShouldClose() const {
  return should_close_;
}

void ModuleBase::RequestClose() {
  should_close_ = true;
}

int ModuleBase::GetFrameIndex() const {
  return frame_index_;
}

void ModuleBase::CompleteFrame() {
  ++frame_index_;
}

void ModuleBase::SetWarmupFrames(int frames) {
  frame_index_ = -frames;
}
