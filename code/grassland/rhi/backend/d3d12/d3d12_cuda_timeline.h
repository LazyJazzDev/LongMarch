#pragma once
#include "grassland/rhi/backend/backend_cuda_timeline.h"
#include "grassland/rhi/backend/d3d12/d3d12_timeline.h"
#include "grassland/rhi/rhi_cuda_timeline.h"
#if defined(LONGMARCH_CUDA_RUNTIME)
namespace grassland::rhi::backend::d3d12 {
ObjectPtr<ID3D12Timeline> CreateD3D12CUDATimeline(ID3D12Device *device);

class D3D12CUDATimelineImpl final : public Object<ICUDATimeline, ID3D12Timeline, IBackendCUDATimeline> {
 public:
  static ObjectPtr<ID3D12Timeline> Create(ID3D12Device *device);
  ~D3D12CUDATimelineImpl() override = default;
  bool GetCompletedValue(uint64_t &value) const override;
  bool WaitHost(uint64_t value) const override;
  ID3D12Fence *NativeFence() const override;

  ID3D12Device *Owner() const override {
    return owner_;
  }

  const void *CUDAOwner() const override {
    return owner_;
  }

  cudaExternalSemaphore_t CUDASemaphore() const override {
    return cuda_.Get();
  }

  bool Wait(cudaStream_t stream, const std::vector<Event> &events) override {
    return cuda_.Wait(owner_, stream, events);
  }

  Event Signal(cudaStream_t stream, uint64_t value) override {
    return cuda_.Signal(stream, value) ? Event{this, value} : Event{};
  }

 private:
  explicit D3D12CUDATimelineImpl(ID3D12Device *device);
  ID3D12Device *owner_;
  ComPtr<ID3D12Fence> fence_;
  CUDAExternalSemaphore cuda_;
};

}  // namespace grassland::rhi::backend::d3d12
#endif
