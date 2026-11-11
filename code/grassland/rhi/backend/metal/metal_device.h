#pragma once
#include <Metal/Metal.hpp>
#include <condition_variable>

#include "grassland/rhi/backend/backend_device.h"
#include "grassland/rhi/backend/metal/metal_scratch_pool.h"
#include "grassland/rhi/backend/metal/metal_temporal_buffer_pool.h"
#include "grassland/rhi/backend/metal/metal_util.h"

LONGMARCH_RHI_UUID(grassland::rhi::backend::metal::IMetalDevice, "0edf854d-4ccc-4b02-8c0c-030ded3b67dc")

namespace grassland::rhi::backend::metal {

class IMetalDevice : public IBackendDevice {
 public:
  static rhi::IDevice *Create(PhysicalDevice info, ICore *core, NS::SharedPtr<MTL::Device> device);
  virtual MetalTemporalBufferPool &UploadPool() = 0;
  virtual MetalTemporalBufferPool &DownloadPool() = 0;
  virtual MetalScratchPool &ScratchPool() = 0;
  virtual MTL::Device *NativeDevice() const noexcept = 0;
  virtual void ObserveCompletion(MTL::CommandBuffer *commands, Event signal) = 0;
  virtual void NotifyTimelineFailure() = 0;
};

}  // namespace grassland::rhi::backend::metal
