#pragma once
#include "grassland/rhi/backend/backend_window.h"
#include "grassland/rhi/backend/vulkan/vulkan_device.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::vulkan::IVulkanWindow)

namespace grassland::rhi::backend::vulkan {

class IVulkanWindow : public IBackendWindow {
 public:
  static rhi::IWindow *Create(IVulkanDevice *device, const WindowDesc &desc);

  virtual VkSemaphore RenderSemaphore() const = 0;
  virtual VkResult PresentFrame() = 0;
};

}  // namespace grassland::rhi::backend::vulkan
