#pragma once
#include <Metal/Metal.hpp>

#include "grassland/rhi/backend/backend_device.h"
#include "grassland/rhi/backend/metal/metal_scratch_pool.h"
#include "grassland/rhi/backend/metal/metal_temporal_buffer_pool.h"

namespace grassland::rhi::backend::metal {

class MetalDevice final : public Device {
 public:
  static MetalDevice *Create(PhysicalDevice info, Core *core, NS::SharedPtr<MTL::Device> device);
  ~MetalDevice() override;

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
  MTL::CommandQueue *NativeQueue(QueueType type = QUEUE_TYPE_GRAPHICS) const;
  NS::SharedPtr<MTL::CommandBuffer> CreateCommandBuffer(QueueType type = QUEUE_TYPE_GRAPHICS);
  std::mutex &QueueMutex(QueueType type = QUEUE_TYPE_GRAPHICS);
  bool WaitIdle();
  ICommandContext *CreateContextForQueue(QueueType type) override;
  bool SharesNativeQueue(QueueType a, QueueType b) const override;

  bool HasNativeTimelines() const override {
    return true;
  }

  bool QueryTimeline(QueueType type, uint64_t &value) const override;
  bool WaitTimelines(const std::vector<TimelinePoint> &points, bool wait_all) const override;
  void WakeCollector() override;
  MTL::SharedEvent *Timeline(QueueType type) const;
  uint64_t NextSerial(QueueType type);
  void TrackCompletion(MTL::CommandBuffer *buffer, QueueType type);

  rhi::IWindow *CreateGLFWWindow(const WindowDesc &desc = {}) override;
  rhi::ICommandContext *CreateCommandContext() override;
  rhi::IQueue *GetQueue(QueueType type) override;

 private:
  MetalDevice(PhysicalDevice info, Core *core, NS::SharedPtr<MTL::Device> device);

  NS::SharedPtr<MTL::Device> device_;
  NS::SharedPtr<MTL::CommandQueue> queue_;
  NS::SharedPtr<MTL::CommandQueue> transfer_queue_;
  std::array<NS::SharedPtr<MTL::SharedEvent>, 2> timelines_;
  std::array<uint64_t, 2> serials_{};
  std::mutex transfer_mutex_;
  mutable std::mutex progress_mutex_;
  mutable std::condition_variable progress_changed_;
  std::array<bool, 2> failed_{};
  NS::SharedPtr<MTL::CommandBufferDescriptor> command_buffer_descriptor_;
  std::mutex queue_mutex_;
  // Shared storage is reused across uploads and downloads.
  MetalTemporalBufferPool temporal_buffers_;
  MetalScratchPool scratch_{this};
};

}  // namespace grassland::rhi::backend::metal
