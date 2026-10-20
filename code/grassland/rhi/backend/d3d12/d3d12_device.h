#pragma once
#include <tuple>

#include "grassland/rhi/backend/backend_device.h"
#include "grassland/rhi/backend/d3d12/d3d12_rtv_pool.h"
#include "grassland/rhi/backend/d3d12/d3d12_upload_pool.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"

namespace grassland::rhi::backend::d3d12 {
class D3D12Device final : public Device {
 public:
  static D3D12Device *Create(PhysicalDevice info, Core *core, ComPtr<ID3D12Device> device);
  ~D3D12Device() override;
  ID3D12Device *NativeDevice() const;
  D3D12MA::Allocator *Allocator() const;
  D3D12RTVPool &RTVPool();
  D3D12DescriptorPool &DSVPool();
  D3D12DescriptorPool &ResourceDescriptors();
  D3D12DescriptorPool &SamplerDescriptors();
  D3D12DescriptorPool &ResourceDescriptorStorage();
  D3D12DescriptorPool &SamplerDescriptorStorage();
  IDXGIFactory4 *Factory() const;
  ID3D12CommandQueue *NativeQueue() const;
  ID3D12Fence *Fence() const;
  bool SupportsTearing() const;
  std::mutex &QueueMutex();
  uint64_t NextSerial();
  bool Wait(uint64_t value, HANDLE event) const;
  bool WaitIdle();
  rhi::IProgram *CreateProgram(const ProgramDesc &desc) override;
  rhi::IImage *CreateImage(const ImageDesc &desc) override;
  rhi::IBindingGroup *CreateBindingGroup(const BindingLayoutDesc &layout) override;
  rhi::IBuffer *CreateBuffer(const BufferDesc &desc) override;
  D3D12UploadPool &UploadPool();
  rhi::IWindow *CreateGLFWWindow(const WindowDesc &desc = {}) override;
  rhi::ICommandContext *CreateCommandContext() override;
  rhi::IQueue *GetQueue(QueueType type) override;

  // Transfer idle native resources; callers recycle only after GPU completion and final handle release.
  // The pool grows on demand, independently of swapchain image count.
  bool AcquireCommandResources(ComPtr<ID3D12CommandAllocator> &allocator,
                               ComPtr<ID3D12GraphicsCommandList> &list,
                               HANDLE &event);
  void RecycleCommandResources(ComPtr<ID3D12CommandAllocator> allocator,
                               ComPtr<ID3D12GraphicsCommandList> list,
                               HANDLE event);

 private:
  D3D12Device(PhysicalDevice info, Core *core, ComPtr<ID3D12Device> device);

  D3D12UploadPool upload_pool_;
  std::mutex command_pool_mutex_;
  std::vector<std::tuple<ComPtr<ID3D12CommandAllocator>, ComPtr<ID3D12GraphicsCommandList>, HANDLE>> command_pool_;
  ComPtr<ID3D12Device> device_;
  ComPtr<D3D12MA::Allocator> allocator_;
  D3D12RTVPool rtv_pool_;
  D3D12DescriptorPool dsv_pool_;
  D3D12DescriptorPool resource_descriptors_, sampler_descriptors_;
  D3D12DescriptorPool resource_descriptor_storage_, sampler_descriptor_storage_;
  ComPtr<IDXGIFactory4> factory_;
  ComPtr<ID3D12CommandQueue> queue_;
  ComPtr<ID3D12Fence> fence_;
  std::mutex mutex_;
  uint64_t serial_{};
  HANDLE idle_event_{};
  bool tearing_{};
};
}  // namespace grassland::rhi::backend::d3d12
