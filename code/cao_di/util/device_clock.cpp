#if defined(CHANGZHENG_CUDA_RUNTIME)

#include "cao_di/util/device_clock.h"

namespace CD {

namespace {
cudaEvent_t CudaCreateAndRecordEvent() {
  cudaEvent_t event;
  cudaEventCreate(&event);
  cudaEventRecord(event);
  return event;
}
}  // namespace

DeviceClock::DeviceClock() {
  cudaDeviceSynchronize();
  start_ = CudaCreateAndRecordEvent();
}

DeviceClock::~DeviceClock() {
  if (start_) {
    cudaEventDestroy(start_);
  }
  if (stop_) {
    cudaEventDestroy(stop_);
  }
  while (!events_.empty()) {
    cudaEventDestroy(events_.front());
    events_.pop();
  }
}

void DeviceClock::Record(const std::string &event_name) {
  events_.push(CudaCreateAndRecordEvent());
  event_names_.push(event_name);
}

void DeviceClock::Finish() {
  stop_ = CudaCreateAndRecordEvent();
  cudaEventSynchronize(stop_);
  float total_time = 0.0f;
  cudaEventElapsedTime(&total_time, start_, stop_);
  printf("[Total time used] %fms\n", total_time);
  float last_time_used = 0.0f;
  while (!events_.empty()) {
    auto event = events_.front();
    events_.pop();
    auto name = event_names_.front();
    event_names_.pop();
    float start_event_dur = 0.0f;
    cudaEventElapsedTime(&start_event_dur, start_, event);
    printf("- [Event] %s, [Time] %fms, [Dur] %fms (%f%%)\n", name.c_str(), start_event_dur,
           start_event_dur - last_time_used, (start_event_dur - last_time_used) / total_time * 100.0f);
    last_time_used = start_event_dur;
    cudaEventDestroy(event);
  }
}

}  // namespace CD

#endif
