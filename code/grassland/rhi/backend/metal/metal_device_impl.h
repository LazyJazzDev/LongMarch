#pragma once
#include "grassland/rhi/backend/backend_device_state.h"
#include "grassland/rhi/backend/metal/metal_device.h"

namespace grassland::rhi::backend::metal {
class MetalDeviceImpl final : public Object<IMetalDevice> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(rhi::IDevice, IMetalDevice, Device, rhi::IDevice)
  static IMetalDevice *Create(PhysicalDevice info, Core *core, NS::SharedPtr<MTL::Device> device);
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
  MetalDeviceImpl(PhysicalDevice info, Core *core, NS::SharedPtr<MTL::Device> device);
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
};
}  // namespace grassland::rhi::backend::metal
