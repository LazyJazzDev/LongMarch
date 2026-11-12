#pragma once
#if defined(LONGMARCH_CUDA_ENABLED)
#include <cuda_runtime_api.h>

#include <iostream>

#include "grassland/rhi/rhi.h"

cudaError_t RunNBodyCUDA(void *positions,
                         void *velocities,
                         void *next_positions,
                         uint32_t count,
                         float delta_t,
                         float gravity,
                         cudaStream_t stream);

// Own the stream and its timeline; finish graphics consumers before releasing either.
class NBodyCUDA final {
 public:
  explicit NBodyCUDA(grassland::rhi::IDevice *device) {
    if (device->GetCUDADeviceIndex() < 0)
      return;
    if (!Check(cudaGetDevice(&previous_device_)) || !Check(cudaSetDevice(device->GetCUDADeviceIndex())))
      return;
    timeline_ = device->CreateCUDATimeline();
    if (timeline_)
      Check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
  }

  NBodyCUDA(const NBodyCUDA &) = delete;
  NBodyCUDA &operator=(const NBodyCUDA &) = delete;

  ~NBodyCUDA() {
    if (stream_) {
      Check(cudaStreamSynchronize(stream_));
      Check(cudaStreamDestroy(stream_));
    }
    if (last_graphics_.timeline)
      last_graphics_.timeline->WaitHost(last_graphics_.value);
    timeline_.Release();
    if (previous_device_ >= 0)
      Check(cudaSetDevice(previous_device_));
  }

  void GraphicsSubmitted(grassland::rhi::Event event) {
    last_graphics_ = event;
  }

  explicit operator bool() const {
    return stream_ != nullptr;
  }

  grassland::rhi::Event Step(void *positions,
                             void *velocities,
                             void *next_positions,
                             uint32_t count,
                             float delta_t,
                             float gravity,
                             grassland::rhi::Event graphics) {
    if (!stream_ || (graphics.timeline && !timeline_->Wait(stream_, {graphics})) ||
        !Check(RunNBodyCUDA(positions, velocities, next_positions, count, delta_t, gravity, stream_)))
      return {};
    return timeline_->Signal(stream_, ++value_);
  }

 private:
  static bool Check(cudaError_t result) {
    if (result == cudaSuccess)
      return true;
    std::cerr << "nbody CUDA: " << cudaGetErrorString(result) << '\n';
    return false;
  }

  cudaStream_t stream_{};
  grassland::rhi::ObjPtr<grassland::rhi::ICUDATimeline> timeline_;
  grassland::rhi::Event last_graphics_{};
  uint64_t value_{};
  int previous_device_{-1};
};
#endif
