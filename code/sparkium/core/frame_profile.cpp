#include "sparkium/core/frame_profile.h"

namespace sparkium {
FrameProfile::FrameProfile(grassland::rhi::IDevice *device) : device_name(device->Name()) {
}

void FrameProfile::Begin() {
  cpu_ms.clear();
  counters.clear();
  counters["gpu_timestamps"] = 0;
  active = this;
}

void FrameProfile::Finish() {
  active = nullptr;
}

CpuProfileScope::CpuProfileScope(const char *name) : profile_(FrameProfile::active), name_(name) {
  if (profile_)
    start_ = std::chrono::steady_clock::now();
}

CpuProfileScope::~CpuProfileScope() {
  End();
}

void CpuProfileScope::End() {
  if (profile_) {
    profile_->cpu_ms[name_] +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_).count();
    profile_ = nullptr;
  }
}
}  // namespace sparkium
