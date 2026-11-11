#pragma once
#include "grassland/rhi/backend/backend_core.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::vulkan::IVulkanCore,
                        grassland::rhi::IObject,
                        "38b5aac6-db8d-4fe0-86ac-ed06f0b3c970")

namespace grassland::rhi::backend::vulkan {

class IVulkanCore : public IObject {
 public:
  static rhi::ICore *CreateVulkanCore(const CoreDesc &desc);
  virtual VkInstance Instance() const = 0;
};

}  // namespace grassland::rhi::backend::vulkan
