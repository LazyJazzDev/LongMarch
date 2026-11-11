#pragma once
#include "grassland/rhi/backend/backend_window_state.h"
#include "grassland/rhi/backend/vulkan/vulkan_image_views.h"
#include "grassland/rhi/backend/vulkan/vulkan_window.h"

namespace grassland::rhi::backend::vulkan {
class VulkanWindowImpl final : public Object<IVulkanWindow> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(rhi::IWindow, IVulkanWindow, Window, rhi::IWindow)
  ~VulkanWindowImpl() override;
  static IVulkanWindow *Create(IVulkanDevice *device, const WindowDesc &desc);
  ObjectPtr<Image> AcquireNativeFrame() override;
  bool RebuildNativeSwapChain(PresentMode mode) override;
  PresentMode GetPresentMode() const override;
  std::vector<WindowPresentationOption> GetSupportedPresentationOptions() const override;
  VkSemaphore RenderSemaphore() const;
  VkResult PresentFrame();

 private:
  VulkanWindowImpl(IVulkanDevice *device);
  void DestroySwapchain();
  IVulkanDevice *device_;
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
  std::vector<VulkanImageViews> image_views_;
  std::vector<VkSemaphore> rendered_;
  std::vector<bool> initialized_images_;
  ImageDesc desc_;
  PresentMode mode_{PRESENT_MODE_VSYNC};

 private:
  WindowState state_Window_;

  WindowState &StateWindow() override {
    return state_Window_;
  }

  const WindowState &StateWindow() const override {
    return state_Window_;
  }
};
}  // namespace grassland::rhi::backend::vulkan
