#pragma once
#include "grassland/rhi/backend/backend_window.h"
#include "grassland/rhi/backend/vulkan/vulkan_device.h"

namespace grassland::rhi::backend::vulkan {

class VulkanWindow final : public Window {
 public:
  bool SupportsImageDownload() const {
    return image_download_supported_;
  }

  ~VulkanWindow() override;
  static VulkanWindow *Create(VulkanDevice *device, const WindowDesc &desc);
  ObjectPtr<Image> AcquireNativeFrame() override;
  bool RebuildNativeSwapChain(PresentMode mode) override;
  PresentMode GetPresentMode() const override;
  std::vector<WindowPresentationOption> GetSupportedPresentationOptions() const override;
  const void *DeviceIdentity() const;
  VkSemaphore AcquireSemaphore() const;
  void MarkAcquireWaitSubmitted(uint64_t value);
  VkSemaphore RenderSemaphore() const;
  VkResult PresentFrame();

 private:
  bool image_download_supported_{};
  VulkanWindow(VulkanDevice *device);

  void DestroySwapchain();
  VulkanDevice *device_;
  VkSurfaceKHR surface_{};
  VkSwapchainKHR swapchain_{};

  // Swapchain-local rotation, independent of which image Acquire returns.
  struct AcquireSync {
    VkSemaphore semaphore{};
    VkFence fence{};
    bool acquired{};
    // Timeline value of the submission consuming this acquire semaphore.
    uint64_t submission_value{};
  };

  std::vector<AcquireSync> acquire_sync_;
  uint32_t current_sync_{};
  uint32_t current_image_{};
  std::vector<VkImage> images_;
  std::vector<VulkanImage::Views> image_views_;
  std::vector<VkSemaphore> rendered_;
  std::vector<bool> initialized_images_;
  ImageDesc desc_;
  PresentMode mode_{PRESENT_MODE_VSYNC};
};

}  // namespace grassland::rhi::backend::vulkan
