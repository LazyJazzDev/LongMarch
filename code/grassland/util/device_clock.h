#pragma once
#if defined(LONGMARCH_CUDA_RUNTIME)
#include <cuda_runtime.h>

#include <queue>
#include <string>

namespace grassland {
class DeviceClock {
 public:
  DeviceClock();
  void Record(const std::string &event_name = "Event");
  void Finish();
  ~DeviceClock();

 private:
  cudaEvent_t start_{}, stop_{};
  std::queue<cudaEvent_t> events_;
  std::queue<std::string> event_names_;
};
}  // namespace grassland

#endif
