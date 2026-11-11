#pragma once
#include "grassland/rhi/backend/backend_compute_program.h"
#include "grassland/rhi/backend/vulkan/vulkan_program_base.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::vulkan::IVulkanComputeProgram)

namespace grassland::rhi::backend::vulkan {
class IVulkanComputeProgram : public IBackendComputeProgram, public IVulkanProgramBase {
 public:
  virtual void AddRef() noexcept override = 0;
  virtual void Release() noexcept override = 0;
  virtual bool QueryInterface(InterfaceID id, void **output) noexcept override = 0;

  template <typename T>
  T *QueryInterface() noexcept {
    void *result = nullptr;
    return QueryInterface(uuid<T>(), &result) ? static_cast<T *>(result) : nullptr;
  }

  static IVulkanComputeProgram *Create(IVulkanDevice *device, const ComputeProgramDesc &desc);
  virtual VkPipeline NativePipeline() const = 0;
};

}  // namespace grassland::rhi::backend::vulkan
