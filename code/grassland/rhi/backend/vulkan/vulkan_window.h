#pragma once
#include "grassland/rhi/backend/backend_window.h"
#include "grassland/rhi/backend/vulkan/vulkan_device.h"

namespace grassland::rhi::backend::vulkan {

class VulkanWindow final : public Window {
 public:
  bool EnableImGui(bool enabled = true, const char *font_file_path = nullptr, float font_size = 13.0f) override {
    return ConfigureImGui(enabled, font_file_path, font_size);
  }

  bool UpdateImGui(const std::function<void()> &callback) override {
    return RunImGuiFrame(callback);
  }

  EventManager<void(int, int)> &ResizeEvent() override {
    return ResizeCallbacks();
  }

  EventManager<void(int, int)> &FramebufferResizeEvent() override {
    return FramebufferResizeCallbacks();
  }

  EventManager<void(bool)> &CursorEnterEvent() override {
    return CursorEnterCallbacks();
  }

  EventManager<void(bool)> &FocusEvent() override {
    return FocusCallbacks();
  }

  EventManager<void(double, double)> &MouseMoveEvent() override {
    return MouseMoveCallbacks();
  }

  EventManager<void(int, int, int, double, double)> &MouseButtonEvent() override {
    return MouseButtonCallbacks();
  }

  EventManager<void(double, double)> &ScrollEvent() override {
    return ScrollCallbacks();
  }

  EventManager<void(int, int, int, int)> &KeyEvent() override {
    return KeyCallbacks();
  }

  EventManager<void(uint32_t)> &CharEvent() override {
    return CharCallbacks();
  }

  EventManager<void(int, const char **)> &DropEvent() override {
    return DropCallbacks();
  }

  rhi::IImage *GetNextImage() override {
    return PreparedImage();
  }

  bool IsSwapChainInvalid() const override {
    return SwapChainNeedsRebuild();
  }

  bool RebuildSwapChain() override {
    return RecreateSwapChain();
  }

  bool SetPresentMode(PresentMode mode) override {
    return ChangePresentMode(mode);
  }

  bool SetColorSpace(ColorSpace color_space) override {
    return ChangeColorSpace(color_space);
  }

  ColorSpace GetColorSpace() const override {
    return ActiveColorSpace();
  }

  bool GetHDRReferenceWhiteNits(float &nits) const override {
    return QueryHDRReferenceWhiteNits(nits);
  }

  float GetHDRReferenceWhiteScale() const override {
    return HDRReferenceWhiteScale();
  }

  float GetHDRHeadroom() const override {
    return HDRHeadroom();
  }

  GLFWwindow *GLFWWindow() const override {
    return NativeWindow();
  }

  bool ShouldClose() const override {
    return WindowShouldClose();
  }

  bool IsKeyDown(int key) const override {
    return KeyPressed(key);
  }

  bool SetTitle(const std::string &title) override {
    return ChangeTitle(title);
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
