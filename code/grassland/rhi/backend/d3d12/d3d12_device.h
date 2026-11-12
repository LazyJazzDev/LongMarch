#pragma once
#include <d3d12sdklayers.h>

#include <array>
#include <atomic>
#include <mutex>
#include <thread>
#include <tuple>

#include "grassland/rhi/backend/d3d12/d3d12_descriptor_pool.h"
#include "grassland/rhi/backend/d3d12/d3d12_scratch_pool.h"
#include "grassland/rhi/backend/d3d12/d3d12_temporal_buffer_pool.h"
#include "grassland/rhi/backend/d3d12/d3d12_transient_descriptor_pool.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"
#include "grassland/rhi/rhi_device.h"

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::d3d12::ID3D12Device,
                        grassland::rhi::IObject,
                        "d258bd31-2295-4f22-9c98-bd0133662c47")

namespace grassland::rhi::backend::d3d12 {
class ID3D12Device : public IObject {
 public:
  virtual const PhysicalDevice &GetPhysicalDevice() const noexcept = 0;
  // Borrowed queue owned by this device.
  virtual ID3D12Queue *GetNativeQueue(QueueType type) const = 0;
  static rhi::IDevice *Create(PhysicalDevice info, ICore *core, ComPtr<::ID3D12Device> device);
  virtual ::ID3D12Device *NativeDevice() const = 0;
  virtual D3D12MA::Allocator *Allocator() const = 0;
  virtual D3D12DescriptorPool &RTVPool() = 0;
  virtual D3D12DescriptorPool &DSVPool() = 0;
  virtual D3D12TransientDescriptorPool &ResourceDescriptors() = 0;
  virtual D3D12TransientDescriptorPool &SamplerDescriptors() = 0;
  virtual D3D12DescriptorPool &ResourceDescriptorStorage() = 0;
  virtual D3D12DescriptorPool &SamplerDescriptorStorage() = 0;
  virtual IDXGIFactory4 *Factory() const = 0;
  virtual bool SupportsTearing() const = 0;
  virtual D3D12TemporalBufferPool &UploadPool() = 0;
  virtual D3D12TemporalBufferPool &DownloadPool() = 0;
  virtual D3D12ScratchPool &ScratchPool() = 0;
  virtual  // Transfer idle native resources; callers recycle only after GPU completion and final handle release.
      // The pool grows on demand, independently of swapchain image count.
      bool
      AcquireCommandResources(ComPtr<ID3D12CommandAllocator> &allocator,
                              ComPtr<ID3D12GraphicsCommandList> &list,
                              D3D12_COMMAND_LIST_TYPE type) = 0;
  virtual void RecycleCommandResources(ComPtr<ID3D12CommandAllocator> allocator,
                                       ComPtr<ID3D12GraphicsCommandList> list) = 0;
};

}  // namespace grassland::rhi::backend::d3d12

namespace grassland::rhi::backend::d3d12 {
class D3D12DeviceImpl final : public Object<rhi::IDevice, ID3D12Device> {
 public:
  ID3D12Queue *GetNativeQueue(QueueType type) const override;
  IRayTracingProgram *CreateRayTracingProgram(const RayTracingProgramDesc &desc) override;
  IBottomLevelAccelerationStructure *CreateBottomLevelAccelerationStructure(
      const BottomLevelAccelerationStructureDesc &desc) override;
  ITopLevelAccelerationStructure *CreateTopLevelAccelerationStructure(
      const TopLevelAccelerationStructureDesc &desc) override;
  static rhi::IDevice *Create(PhysicalDevice info, ICore *core, ComPtr<::ID3D12Device> device);
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

  int GetCUDADeviceIndex() const override {
    return info_.cuda_device;
  }
#if defined(LONGMARCH_CUDA_RUNTIME)
  rhi::ICUDABuffer *CreateCUDABuffer(const BufferDesc &desc) override;
  rhi::ICUDATimeline *CreateCUDATimeline() override;
#endif
  D3D12TemporalBufferPool &UploadPool();
  D3D12TemporalBufferPool &DownloadPool();
  D3D12ScratchPool &ScratchPool();
  rhi::IWindow *CreateGLFWWindow(const WindowDesc &desc = {}) override;
  // Transfer idle native resources; callers recycle only after GPU completion and final handle release.
  // The pool grows on demand, independently of swapchain image count.
  bool AcquireCommandResources(ComPtr<ID3D12CommandAllocator> &allocator,
                               ComPtr<ID3D12GraphicsCommandList> &list,
                               D3D12_COMMAND_LIST_TYPE type);
  void RecycleCommandResources(ComPtr<ID3D12CommandAllocator> allocator, ComPtr<ID3D12GraphicsCommandList> list);

 private:
  void InterruptCollector();
  bool WaitNativeEvents(const std::vector<Event> &events, bool wait_all) const;
  void WaitForProgress(const std::vector<Event> &next);
  D3D12DeviceImpl(PhysicalDevice info, ICore *core, ComPtr<::ID3D12Device> device);
  D3D12TemporalBufferPool upload_pool_;
  D3D12TemporalBufferPool download_pool_;
  D3D12ScratchPool scratch_pool_;
  std::mutex command_pool_mutex_;
  std::array<std::vector<std::tuple<ComPtr<ID3D12CommandAllocator>, ComPtr<ID3D12GraphicsCommandList>>>, 3>
      command_pool_;
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
  std::array<HANDLE, 3> progress_events_{};
  std::array<uint64_t, 3> progress_values_{};

  void InitializeDevice(PhysicalDevice info, ICore *core, BackendAPI api = BACKEND_API_DEFAULT);

 public:
  rhi::IShader *CreateShader(const ShaderDesc &desc) override;

 private:
  void DestroyDevice();

 public:
  std::string Name() const override;

  bool SupportRayTracingPipeline() const override;

  bool SupportRayQuery() const override;

  bool SupportGeometryShader() const override;

  rhi::IQueue *GetQueue(QueueType type) override;

  const PhysicalDevice &GetPhysicalDevice() const noexcept override;

 private:
  ICore *OwnerCore() const;

  // Called once after native device resources and all queues are initialized.
  void StartCollector();

 public:
  bool WaitEvents(const std::vector<Event> &events, bool wait_all = true) const override;

 private:
  void Collect();

  void ShutdownQueues();

  std::array<ObjectPtr<ID3D12Queue>, 3> queues_{};
  std::atomic<bool> stopping_{};
  BackendAPI api_{};
  ICore *core_{};
  PhysicalDevice info_{};
  std::thread collector_{};
};
}  // namespace grassland::rhi::backend::d3d12
