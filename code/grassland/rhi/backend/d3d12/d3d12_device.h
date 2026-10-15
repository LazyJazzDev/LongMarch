#pragma once
#include "grassland/rhi/backend/backend_device.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"

namespace grassland::rhi::backend::d3d12 {
class D3D12Device final : public Device {
 public:
  D3D12Device(PhysicalDevice info, Core *core, ComPtr<ID3D12Device> device);
  ~D3D12Device() override;
  bool IsInitialized() const;
  ID3D12Device *NativeDevice() const;
  IDXGIFactory4 *Factory() const;
  ID3D12CommandQueue *NativeQueue() const;
  ID3D12Fence *Fence() const;
  bool SupportsTearing() const;
  std::mutex &QueueMutex();
  uint64_t NextSerial();
  bool Wait(uint64_t value, HANDLE event) const;
  bool WaitIdle();
  rhi::IWindow *CreateGLFWWindow(const WindowDesc &desc = {}) override;
  rhi::ICommandContext *CreateCommandContext() override;
  rhi::IQueue *GetQueue(QueueType type) override;

 private:
  bool Initialize();
  ComPtr<ID3D12Device> device_;
  ComPtr<IDXGIFactory4> factory_;
  ComPtr<ID3D12CommandQueue> queue_;
  ComPtr<ID3D12Fence> fence_;
  std::mutex mutex_;
  uint64_t serial_{};
  HANDLE idle_event_{};
  bool tearing_{};
  bool initialized_{};
};
}  // namespace grassland::rhi::backend::d3d12
