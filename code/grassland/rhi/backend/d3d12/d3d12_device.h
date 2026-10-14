#pragma once
#include "grassland/rhi/backend/backend_device.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"

namespace grassland::rhi::backend::d3d12 {

class D3D12Device final : public IDevice {
 public:
  // The native lifetime is separate from Device's submission collector so that
  // in-flight images/commands cannot keep their owning collector alive in a cycle.
  class BackendLifetime {
   public:
    explicit BackendLifetime(ComPtr<ID3D12Device> device);
    ~BackendLifetime();
    bool Initialize();
    ID3D12Device *Device() const;
    IDXGIFactory4 *Factory() const;
    ID3D12CommandQueue *Queue() const;
    ID3D12Fence *Fence() const;
    bool SupportsTearing() const;
    std::mutex &QueueMutex();
    uint64_t NextSerial();
    bool Wait(uint64_t value, HANDLE event) const;
    bool WaitIdle();

   private:
    ComPtr<ID3D12Device> device_;
    ComPtr<IDXGIFactory4> factory_;
    ComPtr<ID3D12CommandQueue> queue_;
    ComPtr<ID3D12Fence> fence_;
    std::mutex mutex_;
    uint64_t serial_{};
    HANDLE idle_event_{};
    bool tearing_{};
  };

  D3D12Device(PhysicalDevice info, const ObjectPtr<ICore> &core, ComPtr<ID3D12Device> device);
  ~D3D12Device() override;
  bool IsInitialized() const;
  const std::shared_ptr<BackendLifetime> &NativeLifetime() const;
  const void *NativeIdentity() const override;
  std::unique_ptr<IWindow> CreateWindow(const WindowDesc &desc) override;
  std::unique_ptr<IQueue> CreateQueue(QueueType type) override;

 private:
  std::shared_ptr<BackendLifetime> lifetime_;
};

}  // namespace grassland::rhi::backend::d3d12
