#pragma once
#include "grassland/rhi/backend/backend_device_state.h"
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/backend/metal/metal_device.h"

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

 private:
  DeviceState state_Device_;

  DeviceState &StateDevice() override {
    return state_Device_;
  }

  const DeviceState &StateDevice() const override {
    return state_Device_;
  }

 public:
  IRayTracingProgram *CreateRayTracingProgram(const RayTracingProgramDesc &desc) override {
    return IBackendDevice::CommonCreateRayTracingProgram(desc);
  }

  std::string Name() const override {
    return IBackendDevice::CommonName();
  }

  bool SupportRayTracingPipeline() const override {
    return IBackendDevice::CommonSupportRayTracingPipeline();
  }

  bool SupportRayQuery() const override {
    return IBackendDevice::CommonSupportRayQuery();
  }

  bool SupportGeometryShader() const override {
    return IBackendDevice::CommonSupportGeometryShader();
  }

  rhi::IShader *CreateShader(const ShaderDesc &desc) override {
    return IBackendDevice::CommonCreateShader(desc);
  }

  rhi::IQueue *GetQueue(QueueType type) override {
    return IBackendDevice::CommonGetQueue(type);
  }

  bool WaitEvents(const std::vector<Event> &events, bool wait_all = true) const override {
    return IBackendDevice::CommonWaitEvents(events, wait_all);
  }
};
}  // namespace grassland::rhi::backend::metal
