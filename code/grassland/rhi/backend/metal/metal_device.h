#pragma once
#include <Metal/Metal.hpp>
#include <array>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>

#include "grassland/rhi/backend/metal/metal_scratch_pool.h"
#include "grassland/rhi/backend/metal/metal_temporal_buffer_pool.h"
#include "grassland/rhi/backend/metal/metal_util.h"
#include "grassland/rhi/rhi_device.h"

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::metal::IMetalDevice,
                        grassland::rhi::IObject,
                        "0edf854d-4ccc-4b02-8c0c-030ded3b67dc")

namespace grassland::rhi::backend::metal {

class IMetalDevice : public IObject {
 public:
  virtual const PhysicalDevice &GetPhysicalDevice() const noexcept = 0;
  // Borrowed queue owned by this device.
  virtual IMetalQueue *GetNativeQueue(QueueType type) const = 0;
  static rhi::IDevice *Create(PhysicalDevice info, IMetalCore *core, NS::SharedPtr<MTL::Device> device);
  virtual MetalTemporalBufferPool &UploadPool() = 0;
  virtual MetalTemporalBufferPool &DownloadPool() = 0;
  virtual MetalScratchPool &ScratchPool() = 0;
  virtual MTL::Device *NativeDevice() const noexcept = 0;
  virtual void ObserveCompletion(MTL::CommandBuffer *commands, Event signal) = 0;
  virtual void NotifyTimelineFailure() = 0;
};

}  // namespace grassland::rhi::backend::metal

namespace grassland::rhi::backend::metal {
class MetalDeviceImpl final : public Object<rhi::IDevice, IMetalDevice> {
 public:
  IMetalQueue *GetNativeQueue(QueueType type) const override;
  static rhi::IDevice *Create(PhysicalDevice info, IMetalCore *core, NS::SharedPtr<MTL::Device> device);
  ~MetalDeviceImpl() override;
  rhi::IBuffer *CreateBuffer(const BufferDesc &desc) override;

  int GetCUDADeviceIndex() const override {
    return info_.cuda_device;
  }
#if defined(LONGMARCH_CUDA_RUNTIME)
  rhi::ICUDABuffer *CreateCUDABuffer(const BufferDesc &desc) override;
  rhi::ICUDATimeline *CreateCUDATimeline() override;
#endif
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
  void InterruptCollector();
  bool WaitNativeEvents(const std::vector<Event> &events, bool wait_all) const;
  void WaitForProgress(const std::vector<Event> &next);
  MetalDeviceImpl(PhysicalDevice info, NS::SharedPtr<MTL::Device> device);
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

 public:
  rhi::IShader *CreateShader(const ShaderDesc &desc) override;

 public:
  std::string Name() const override;

  bool SupportRayTracingPipeline() const override;

  bool SupportRayQuery() const override;

  bool SupportGeometryShader() const override;

  rhi::IQueue *GetQueue(QueueType type) override;

  const PhysicalDevice &GetPhysicalDevice() const noexcept override;

 private:
  // Called once after native device resources and all queues are initialized.
  void StartCollector();

 public:
  bool WaitEvents(const std::vector<Event> &events, bool wait_all = true) const override;

 private:
  void Collect();

  void ShutdownQueues();

  std::array<ObjectPtr<IMetalQueue>, 3> queues_{};
  std::atomic<bool> stopping_{};
  PhysicalDevice info_{};
  std::thread collector_{};
};
}  // namespace grassland::rhi::backend::metal
