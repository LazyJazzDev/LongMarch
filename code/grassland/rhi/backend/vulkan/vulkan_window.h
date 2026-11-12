#pragma once
#include "grassland/rhi/backend/backend_window.h"
#include "grassland/rhi/backend/vulkan/vulkan_device.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::vulkan::IVulkanWindow,
                        grassland::rhi::backend::IBackendWindow,
                        "2bd587e1-b088-4f5c-8138-3fd885407013")

namespace grassland::rhi::backend::vulkan {

class IVulkanWindow : public IBackendWindow {
 public:
  virtual IVulkanQueue *OwnerQueue() const = 0;
  virtual bool RecordImGui(rhi::ICommandContext *commands, rhi::IImage *image) = 0;
  virtual bool RecordImGui(rhi::ICommandContext *commands,
                           rhi::IImage *image,
                           ColorSpace color_space,
                           float white_scale) = 0;
  virtual bool RecordPresentImage(ICommandContext *commands, IImage *image, ColorSpace color_space) = 0;
  virtual const void *DeviceIdentity() const = 0;
  virtual bool UpdateNextImage() = 0;
  virtual void ReleaseNextImage() = 0;
  virtual ObjectPtr<IVulkanImage> AcquireNativeFrame() = 0;
  virtual bool RebuildNativeSwapChain(PresentMode mode) = 0;
  static rhi::IWindow *Create(IVulkanDevice *device, const WindowDesc &desc);

  virtual VkSemaphore RenderSemaphore() const = 0;
  virtual VkResult PresentFrame() = 0;
};

}  // namespace grassland::rhi::backend::vulkan
