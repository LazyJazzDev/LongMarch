#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_interface_id.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::metal::IMetalDevice)
#include <Metal/Metal.hpp>
#include <condition_variable>

#include "grassland/rhi/backend/backend_device.h"
#include "grassland/rhi/backend/metal/metal_scratch_pool.h"
#include "grassland/rhi/backend/metal/metal_temporal_buffer_pool.h"

namespace grassland::rhi::backend::metal {

class IMetalDevice : public IBackendDevice {
 public:
  static IMetalDevice *Create(PhysicalDevice info, IBackendCore *core, NS::SharedPtr<MTL::Device> device);
  virtual rhi::IBuffer *CreateBuffer(const BufferDesc &desc) = 0;
  virtual rhi::IImage *CreateImage(const ImageDesc &desc) = 0;
  virtual IComputeProgram *CreateComputeProgram(const ComputeProgramDesc &desc) = 0;
  virtual rhi::IProgram *CreateProgram(const ProgramDesc &desc) = 0;
  virtual rhi::IBindingGroup *CreateBindingGroup(const BindingLayoutDesc &layout) = 0;
  virtual ISampler *CreateSampler(const SamplerDesc &desc = {}) = 0;
  virtual MetalTemporalBufferPool &UploadPool() = 0;
  virtual MetalTemporalBufferPool &DownloadPool() = 0;
  virtual MetalScratchPool &ScratchPool() = 0;
  virtual IBottomLevelAccelerationStructure *CreateBottomLevelAccelerationStructure(
      const BottomLevelAccelerationStructureDesc &desc) = 0;
  virtual ITopLevelAccelerationStructure *CreateTopLevelAccelerationStructure(
      const TopLevelAccelerationStructureDesc &desc) = 0;
  virtual MTL::Device *NativeDevice() const noexcept = 0;
  virtual rhi::IWindow *CreateGLFWWindow(const WindowDesc &desc = {}) = 0;
  virtual void ObserveCompletion(MTL::CommandBuffer *commands, Event signal) = 0;
  virtual void NotifyTimelineFailure() = 0;
};

;

}  // namespace grassland::rhi::backend::metal
