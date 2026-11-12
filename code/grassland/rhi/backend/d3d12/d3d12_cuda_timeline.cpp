#include "grassland/rhi/backend/d3d12/d3d12_cuda_timeline.h"

#include "grassland/rhi/backend/backend_cuda_timeline.h"
#if defined(LONGMARCH_CUDA_RUNTIME)

#include "grassland/rhi/backend/d3d12/d3d12_device.h"

namespace grassland::rhi::backend::d3d12 {

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

D3D12CUDATimelineImpl::D3D12CUDATimelineImpl(ID3D12Device *device)
    : owner_(device),
      cuda_(device->GetPhysicalDevice().cuda_device) {
}

ObjectPtr<ID3D12Timeline> D3D12CUDATimelineImpl::Create(ID3D12Device *device) {
  if (device->GetPhysicalDevice().cuda_device < 0)
    return {};
  auto result = ObjectPtr<D3D12CUDATimelineImpl>::Adopt(new D3D12CUDATimelineImpl(device));
  if (!Check(device->NativeDevice()->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&result->fence_)),
             "Create queue timeline"))
    return {};

  HANDLE handle{};
  if (!Check(device->NativeDevice()->CreateSharedHandle(result->fence_.Get(), nullptr, GENERIC_ALL, nullptr, &handle),
             "Export CUDA timeline"))
    return {};
  cudaExternalSemaphoreHandleDesc desc{};
  desc.type = cudaExternalSemaphoreHandleTypeD3D12Fence;
  desc.handle.win32.handle = handle;
  const bool imported = result->cuda_.Import(desc);
  CloseHandle(handle);
  if (!imported)
    return {};

  return result;
}

bool D3D12CUDATimelineImpl::GetCompletedValue(uint64_t &value) const {
  value = fence_->GetCompletedValue();
  return value != UINT64_MAX;
}

bool D3D12CUDATimelineImpl::WaitHost(uint64_t value) const {
  uint64_t completed;
  if (!GetCompletedValue(completed))
    return false;
  if (completed >= value)
    return true;
  // A null event requests the native blocking wait without a separate Win32 handle.
  return Check(fence_->SetEventOnCompletion(value, nullptr), "Wait for timeline") && GetCompletedValue(completed) &&
         completed >= value;
}

ID3D12Fence *D3D12CUDATimelineImpl::NativeFence() const {
  return fence_.Get();
}

ObjectPtr<ID3D12Timeline> CreateD3D12CUDATimeline(ID3D12Device *device) {
  return D3D12CUDATimelineImpl::Create(device);
}
}  // namespace grassland::rhi::backend::d3d12

#endif
