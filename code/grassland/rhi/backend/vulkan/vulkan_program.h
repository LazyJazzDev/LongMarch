#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_interface_id.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::vulkan::IVulkanProgram)
#include "grassland/rhi/backend/backend_program.h"
#include "grassland/rhi/backend/vulkan/vulkan_program_base.h"

namespace grassland::rhi::backend::vulkan {
class IVulkanProgram : public IBackendProgram, public IVulkanProgramBase {
 public:
  virtual void AddRef() noexcept override = 0;
  virtual void Release() noexcept override = 0;
  virtual bool QueryInterface(InterfaceID id, void **output) noexcept override = 0;

  template <typename T>
  T *QueryInterface() noexcept {
    void *result = nullptr;
    return QueryInterface(uuid<T>(), &result) ? static_cast<T *>(result) : nullptr;
  }

 public:
  static IVulkanProgram *Create(IVulkanDevice *device, const ProgramDesc &desc);
  virtual VkPipeline NativePipeline() const = 0;
};

;
}  // namespace grassland::rhi::backend::vulkan
