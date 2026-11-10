#pragma once
#include <Metal/Metal.hpp>
#include <condition_variable>

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
  rhi::IWindow *CreateGLFWWindow(const WindowDesc &desc = {}) override;

 private:
  bool WaitNativeEvents(const std::vector<Event> &events, bool wait_all) const override;
  void WaitForProgress(const std::vector<Event> &next) override;
  MetalDevice(PhysicalDevice info, Core *core, NS::SharedPtr<MTL::Device> device);

  NS::SharedPtr<MTL::Device> device_;

  struct WaitState {
    std::mutex mutex;
    std::condition_variable changed;
    uint64_t generation{};
    std::array<uint64_t, 2> values{};
    NS::SharedPtr<MTL::SharedEventListener> listener;
  };

  std::shared_ptr<WaitState> wait_state_{std::make_shared<WaitState>()};
  // Shared storage is reused across uploads and downloads.
  MetalTemporalBufferPool temporal_buffers_;
  MetalScratchPool scratch_{this};
};

}  // namespace grassland::rhi::backend::metal
