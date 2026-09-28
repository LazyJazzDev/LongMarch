#pragma once
#include <optional>

#include "grassland/graphics/backend/vulkan/vulkan_core.h"
#include "grassland/graphics/backend/vulkan/vulkan_imgui_assets.h"
#include "grassland/graphics/backend/vulkan/vulkan_util.h"

namespace grassland::graphics::backend {

class VulkanWindow : public Window {
 public:
  VulkanWindow(VulkanCore *core,
               int width,
               int height,
               const std::string &title,
               bool fullscreen,
               bool resizable,
               bool enable_hdr);
  ~VulkanWindow();

  virtual void CloseWindow() override;
  int SetHDR(bool enable_hdr) override;

  // An HDR request must never silently select an SDR format.
  static std::optional<VkSurfaceFormatKHR> ChooseHDRSurfaceFormat(const std::vector<VkSurfaceFormatKHR> &formats);

  VkExtent2D SwapChainExtent() const {
    return swap_chain_extent_;
  }

  VkFormat SwapChainFormat() const {
    return swap_chain_format_;
  }

  VkSurfaceKHR Surface() const {
    return surface_;
  }

  VkPhysicalDevice PhysicalDevice() const {
    return core_->PhysicalDevice();
  }

  VkSemaphore RenderFinishSemaphore() const;

  VkSemaphore ImageAvailableSemaphore() const;

  uint32_t AcquireNextImage();

  int Rebuild();

  void Present();

  uint32_t CurrentImageIndex() const {
    return image_index_;
  }

  VkImage CurrentImage() const {
    return swap_chain_images_[image_index_];
  }

  void InitImGui(const char *font_file_path, float font_size) override;
  void TerminateImGui() override;
  void BeginImGuiFrame() override;
  void EndImGuiFrame() override;
  ImGuiContext *GetImGuiContext() const override;

  VulkanImGuiAssets &ImGuiAssets();
  void SetupImGuiContext();
  void BuildImGuiFramebuffers();
  void DestroyImGuiFramebuffers();
  void DestroyImGuiRenderPass();
  VkFramebuffer HDRFramebuffer(VulkanImage *image);

 protected:
  virtual VkResult CreatePresentationSwapchain(VkSurfaceFormatKHR format,
                                               VkSwapchainKHR *result,
                                               VkSwapchainKHR old_swapchain);

 private:
  bool presentation_failed_{false};
  bool UsesPQOutput() const override;
  std::optional<VkSurfaceFormatKHR> SelectSurfaceFormat(bool hdr) const;
  VkFormat ImGuiFormat() const;
  VkSurfaceFormatKHR surface_format_{};
  VkQueue present_queue_;
  VulkanCore *core_;
  VkSurfaceKHR surface_{VK_NULL_HANDLE};
  VkSwapchainKHR swap_chain_{VK_NULL_HANDLE};
  VkFormat swap_chain_format_{VK_FORMAT_UNDEFINED};
  VkExtent2D swap_chain_extent_{};
  std::vector<VkImage> swap_chain_images_;
  std::vector<VkImageView> swap_chain_image_views_;
  void CreateSwapChain();
  void LoadSwapChainImages();
  void DestroySwapChain();
  std::vector<VkSemaphore> render_finish_semaphores_;
  std::vector<VkSemaphore> image_available_semaphores_;
  uint32_t image_index_;

  VulkanImGuiAssets imgui_assets_{};
  VkFramebuffer hdr_framebuffer_{VK_NULL_HANDLE};
};

}  // namespace grassland::graphics::backend
