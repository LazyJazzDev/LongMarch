#pragma once
#include "grassland/rhi/backend/backend_core.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::vulkan::IVulkanCore)

namespace grassland::rhi::backend::vulkan {

class IVulkanCore : public IObject {
 public:
  static rhi::ICore *CreateVulkanCore(const CoreDesc &desc);
  virtual VkInstance Instance() const = 0;
};

}  // namespace grassland::rhi::backend::vulkan
