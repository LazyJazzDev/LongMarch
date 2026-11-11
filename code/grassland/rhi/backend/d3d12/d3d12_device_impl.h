#pragma once
#include "grassland/rhi/backend/backend_device_state.h"
#include "grassland/rhi/backend/d3d12/d3d12_device.h"

namespace grassland::rhi::backend::d3d12 {
class D3D12DeviceImpl final : public Object<ID3D12Device> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(rhi::IDevice, ID3D12Device, Device, rhi::IDevice)
  IRayTracingProgram *CreateRayTracingProgram(const RayTracingProgramDesc &desc) override;
  IBottomLevelAccelerationStructure *CreateBottomLevelAccelerationStructure(
      const BottomLevelAccelerationStructureDesc &desc) override;
  ITopLevelAccelerationStructure *CreateTopLevelAccelerationStructure(
      const TopLevelAccelerationStructureDesc &desc) override;
  static ID3D12Device *Create(PhysicalDevice info, Core *core, ComPtr<::ID3D12Device> device);
  ~D3D12DeviceImpl() override;
  ::ID3D12Device *NativeDevice() const;
  D3D12MA::Allocator *Allocator() const;
  D3D12DescriptorPool &RTVPool();
  D3D12DescriptorPool &DSVPool();
  D3D12TransientDescriptorPool &ResourceDescriptors();
  D3D12TransientDescriptorPool &SamplerDescriptors();
  D3D12DescriptorPool &ResourceDescriptorStorage();
  D3D12DescriptorPool &SamplerDescriptorStorage();
  IDXGIFactory4 *Factory() const;
  bool SupportsTearing() const;
  rhi::IComputeProgram *CreateComputeProgram(const ComputeProgramDesc &desc) override;
  rhi::IProgram *CreateProgram(const ProgramDesc &desc) override;
  rhi::IImage *CreateImage(const ImageDesc &desc) override;
  ISampler *CreateSampler(const SamplerDesc &desc = {}) override;
  rhi::IBindingGroup *CreateBindingGroup(const BindingLayoutDesc &layout) override;
  rhi::IBuffer *CreateBuffer(const BufferDesc &desc) override;
  D3D12TemporalBufferPool &UploadPool();
  D3D12TemporalBufferPool &DownloadPool();
  D3D12ScratchPool &ScratchPool();
  rhi::IWindow *CreateGLFWWindow(const WindowDesc &desc = {}) override;
  // Transfer idle native resources; callers recycle only after GPU completion and final handle release.
  // The pool grows on demand, independently of swapchain image count.
  bool AcquireCommandResources(ComPtr<ID3D12CommandAllocator> &allocator, ComPtr<ID3D12GraphicsCommandList> &list);
  void RecycleCommandResources(ComPtr<ID3D12CommandAllocator> allocator, ComPtr<ID3D12GraphicsCommandList> list);

 private:
  void InterruptCollector() override;
  bool WaitNativeEvents(const std::vector<Event> &events, bool wait_all) const override;
  void WaitForProgress(const std::vector<Event> &next) override;
  D3D12DeviceImpl(PhysicalDevice info, Core *core, ComPtr<::ID3D12Device> device);
  D3D12TemporalBufferPool upload_pool_;
  D3D12TemporalBufferPool download_pool_;
  D3D12ScratchPool scratch_pool_;
  std::mutex command_pool_mutex_;
  std::vector<std::tuple<ComPtr<ID3D12CommandAllocator>, ComPtr<ID3D12GraphicsCommandList>>> command_pool_;
  ComPtr<::ID3D12Device> device_;
  ComPtr<ID3D12InfoQueue1> debug_messages_;
  DWORD debug_callback_cookie_{};
  ComPtr<D3D12MA::Allocator> allocator_;
  D3D12DescriptorPool rtv_pool_;
  D3D12DescriptorPool dsv_pool_;
  D3D12TransientDescriptorPool resource_descriptors_, sampler_descriptors_;
  D3D12DescriptorPool resource_descriptor_storage_, sampler_descriptor_storage_;
  ComPtr<IDXGIFactory4> factory_;
  bool tearing_{};
  HANDLE collector_shutdown_{};
  std::array<HANDLE, 2> progress_events_{};
  std::array<uint64_t, 2> progress_values_{};

 private:
  DeviceState state_Device_;

  DeviceState &StateDevice() override {
    return state_Device_;
  }

  const DeviceState &StateDevice() const override {
    return state_Device_;
  }
};
}  // namespace grassland::rhi::backend::d3d12
