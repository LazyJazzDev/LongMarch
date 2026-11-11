#pragma once
#include "grassland/rhi/backend/backend_window.h"
#include "grassland/rhi/backend/vulkan/vulkan_device.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::vulkan::IVulkanWindow)

namespace grassland::rhi::backend::vulkan {

class IVulkanWindow : public IBackendWindow {
 public:
  static IVulkanWindow *Create(IVulkanDevice *device, const WindowDesc &desc);
  virtual ObjectPtr<IBackendImage> AcquireNativeFrame() = 0;
  virtual bool RebuildNativeSwapChain(PresentMode mode) = 0;
  virtual PresentMode GetPresentMode() const = 0;
  virtual std::vector<WindowPresentationOption> GetSupportedPresentationOptions() const = 0;
  virtual VkSemaphore RenderSemaphore() const = 0;
  virtual VkResult PresentFrame() = 0;
};

}  // namespace grassland::rhi::backend::vulkan
