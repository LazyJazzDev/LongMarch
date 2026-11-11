#pragma once
#include "grassland/rhi/backend/backend_window.h"
#include "grassland/rhi/backend/vulkan/vulkan_device.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

LONGMARCH_RHI_UUID(grassland::rhi::backend::vulkan::IVulkanWindow, "2bd587e1-b088-4f5c-8138-3fd885407013")

namespace grassland::rhi::backend::vulkan {

class IVulkanWindow : public IBackendWindow {
 public:
  static rhi::IWindow *Create(IVulkanDevice *device, const WindowDesc &desc);

  virtual VkSemaphore RenderSemaphore() const = 0;
  virtual VkResult PresentFrame() = 0;
};

}  // namespace grassland::rhi::backend::vulkan
