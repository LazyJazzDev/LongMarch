#pragma once

#include <chrono>
#include <map>
#include <string>

#include "grassland/rhi/rhi_device.h"

namespace sparkium {
// CPU-only profiling. GPU timestamps are not exposed by RHI yet.
class FrameProfile {
 public:
  explicit FrameProfile(grassland::rhi::IDevice *device);
  void Begin();
  void Finish();

  inline static thread_local FrameProfile *active{};
  std::map<std::string, double> cpu_ms;
  std::map<std::string, uint64_t> counters;
  std::string device_name;
};

class CpuProfileScope {
 public:
  explicit CpuProfileScope(const char *name);
  ~CpuProfileScope();
  void End();

 private:
  FrameProfile *profile_;
  const char *name_;
  std::chrono::steady_clock::time_point start_;
};
}  // namespace sparkium
