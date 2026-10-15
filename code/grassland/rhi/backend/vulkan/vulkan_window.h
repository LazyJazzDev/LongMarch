#pragma once
#include "grassland/rhi/backend/backend_window.h"
#include "grassland/rhi/backend/vulkan/vulkan_device.h"

namespace grassland::rhi::backend::vulkan {

class VulkanWindow final : public Window {
 public:
  explicit VulkanWindow(VulkanDevice *device);
  ~VulkanWindow() override;
  bool OwnsFrame(const Image *image) const override;
  bool Initialize(const WindowDesc &desc);
  ObjectPtr<Image> AcquireNativeFrame() override;
  bool SetNativePresentMode(PresentMode mode) override;
  PresentMode GetPresentMode() const override;
  std::vector<WindowPresentationOption> GetSupportedPresentationOptions() const override;
  bool IsFrameValid(uint64_t generation) const;
  const void *DeviceIdentity() const;
  void CancelFrame(uint64_t generation);
  VkResult AcquireNative(VkSemaphore semaphore, VkFence fence, uint32_t &index, VkImage &image, VkSemaphore &rendered);
  VkResult PresentFrame(uint64_t generation, uint32_t index, VkSemaphore rendered);

 private:
  bool Rebuild();
  void DestroySwapchain();
  VulkanDevice *device_;
  VkSurfaceKHR surface_{};
  VkSwapchainKHR swapchain_{};
  std::vector<VkImage> images_;
  std::vector<VkSemaphore> rendered_;
  VkFence pending_acquire_{};
  ImageDesc desc_;
  std::atomic<uint64_t> generation_{1};
  bool acquired_{};
  bool valid_{};
  PresentMode requested_{PRESENT_MODE_VSYNC};
  PresentMode mode_{PRESENT_MODE_VSYNC};
  int width_{}, height_{};
};

}  // namespace grassland::rhi::backend::vulkan
