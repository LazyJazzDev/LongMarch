#pragma once
#include "grassland/rhi/backend/backend_window.h"
#include "grassland/rhi/backend/vulkan/vulkan_device.h"

namespace grassland::rhi::backend::vulkan {

class VulkanCommandContext;

class VulkanWindow final : public Window {
 public:
  explicit VulkanWindow(VulkanDevice *device);
  ~VulkanWindow() override;
  bool Initialize(const WindowDesc &desc);
  ObjectPtr<Image> AcquireNativeFrame() override;
  bool RebuildNativeSwapChain() override;
  bool SetNativePresentMode(PresentMode mode) override;
  PresentMode GetPresentMode() const override;
  std::vector<WindowPresentationOption> GetSupportedPresentationOptions() const override;
  const void *DeviceIdentity() const;
  void MarkImageInitialized(VkImage image);
  VkSemaphore AcquireSemaphore() const;
  void ConsumeAcquireSemaphore(VulkanCommandContext *submission);
  VkSemaphore RenderSemaphore() const;
  VkResult PresentFrame();

 private:
  VkResult AcquireNative();
  void DestroySwapchain();
  VulkanDevice *device_;
  VkSurfaceKHR surface_{};
  VkSwapchainKHR swapchain_{};

  // Swapchain-local rotation, independent of which image Acquire returns.
  struct AcquireSync {
    VkSemaphore semaphore{};
    VkFence fence{};
    bool acquired{};
    // Pins the submission fence against reset until this record is reused.
    ObjectPtr<VulkanCommandContext> submission;
  };

  std::vector<AcquireSync> acquire_sync_;
  uint32_t next_sync_{};
  uint32_t current_sync_{};
  uint32_t current_image_{};
  bool acquire_wait_pending_{};
  std::vector<VkImage> images_;
  std::vector<VkSemaphore> rendered_;
  std::vector<bool> initialized_images_;
  ImageDesc desc_;
  bool acquired_{};
  bool valid_{};
  PresentMode requested_{PRESENT_MODE_VSYNC};
  PresentMode mode_{PRESENT_MODE_VSYNC};
  int width_{}, height_{};
};

}  // namespace grassland::rhi::backend::vulkan
