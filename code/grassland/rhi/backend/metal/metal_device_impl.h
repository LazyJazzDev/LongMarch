#pragma once
#include "grassland/rhi/backend/metal/metal_device.h"
#include "grassland/rhi/backend/metal/metal_util.h"

namespace grassland::rhi::backend::metal {
class MetalDeviceImpl final : public Object<IMetalDevice> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(rhi::IDevice, IMetalDevice, IBackendDevice, rhi::IDevice)
  static IMetalDevice *Create(PhysicalDevice info, IBackendCore *core, NS::SharedPtr<MTL::Device> device);
  ~MetalDeviceImpl() override;
  rhi::IBuffer *CreateBuffer(const BufferDesc &desc) override;
  rhi::IImage *CreateImage(const ImageDesc &desc) override;
  IComputeProgram *CreateComputeProgram(const ComputeProgramDesc &desc) override;
  rhi::IProgram *CreateProgram(const ProgramDesc &desc) override;
  rhi::IBindingGroup *CreateBindingGroup(const BindingLayoutDesc &layout) override;
  ISampler *CreateSampler(const SamplerDesc &desc = {}) override;
  MetalTemporalBufferPool &UploadPool();
  MetalTemporalBufferPool &DownloadPool();
  MetalScratchPool &ScratchPool();
  IBottomLevelAccelerationStructure *CreateBottomLevelAccelerationStructure(
      const BottomLevelAccelerationStructureDesc &desc) override;
  ITopLevelAccelerationStructure *CreateTopLevelAccelerationStructure(
      const TopLevelAccelerationStructureDesc &desc) override;
  MTL::Device *NativeDevice() const noexcept;
  rhi::IWindow *CreateGLFWWindow(const WindowDesc &desc = {}) override;
  void ObserveCompletion(MTL::CommandBuffer *commands, Event signal);
  void NotifyTimelineFailure();

 private:
  void InterruptCollector() override;
  bool WaitNativeEvents(const std::vector<Event> &events, bool wait_all) const override;
  void WaitForProgress(const std::vector<Event> &next) override;
  MetalDeviceImpl(PhysicalDevice info, IBackendCore *core, NS::SharedPtr<MTL::Device> device);
  NS::SharedPtr<MTL::Device> device_;

  struct WaitState {
    std::mutex mutex;
    std::condition_variable changed;
    bool notified{};
    NS::SharedPtr<MTL::SharedEventListener> listener;
  };

  std::shared_ptr<WaitState> wait_state_{std::make_shared<WaitState>()};
  // Shared storage is reused across uploads and downloads.
  MetalTemporalBufferPool temporal_buffers_;
  MetalScratchPool scratch_{this};

 public:
  IRayTracingProgram *CreateRayTracingProgram(const RayTracingProgramDesc &desc) override;

 private:
  void InitializeDevice(PhysicalDevice info, IBackendCore *core, BackendAPI api = BACKEND_API_DEFAULT);

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
  IBackendCore *OwnerCore() const;

  // Called once after native device resources and all queues are initialized.
  void StartCollector();

 public:
  bool WaitEvents(const std::vector<Event> &events, bool wait_all = true) const override;

 private:
  void Collect();

  void ShutdownQueues();

  std::array<ObjectPtr<IBackendQueue>, 2> queues_{};
  std::atomic<bool> stopping_{};
  BackendAPI api_{};
  IBackendCore *core_{};
  PhysicalDevice info_{};
  std::thread collector_{};
};
}  // namespace grassland::rhi::backend::metal
