#include "grassland/graphics/backend/vulkan/vulkan_window.h"

namespace grassland::graphics::backend {

VulkanWindow::VulkanWindow(VulkanCore *core,
                           int width,
                           int height,
                           const std::string &title,
                           bool fullscreen,
                           bool resizable,
                           bool enable_hdr)
    : Window(width, height, title, fullscreen, resizable, enable_hdr), core_(core) {
  core_->Instance()->CreateSurfaceFromGLFWWindow(GLFWWindow(), &surface_);
  core_->Device()->CreateSwapchain(
      surface_.get(), enable_hdr_ ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R8G8B8A8_UNORM,
      enable_hdr_ ? VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT : VK_COLOR_SPACE_SRGB_NONLINEAR_KHR, &swap_chain_);
  image_available_semaphores_.resize(swap_chain_->ImageCount());
  render_finish_semaphores_.resize(swap_chain_->ImageCount());
  for (size_t i = 0; i < image_available_semaphores_.size(); ++i) {
    core_->Device()->CreateSemaphore(&image_available_semaphores_[i]);
    core_->Device()->CreateSemaphore(&render_finish_semaphores_[i]);
  }
  vkGetDeviceQueue(core_->Device()->Handle(), core_->Device()->PhysicalDevice().PresentFamilyIndex(surface_.get()), 0,
                   &present_queue_);
  ResizeEvent().RegisterCallback([this](int width, int height) { Rebuild(); });
}

void VulkanWindow::CloseWindow() {
  core_->WaitGPU();
  image_available_semaphores_.clear();
  render_finish_semaphores_.clear();
  swap_chain_.reset();
  surface_.reset();
  Window::CloseWindow();
}

vulkan::Swapchain *VulkanWindow::SwapChain() const {
  return swap_chain_.get();
}

vulkan::Semaphore *VulkanWindow::RenderFinishSemaphore() const {
  return render_finish_semaphores_[core_->CurrentFrame()].get();
}

vulkan::Semaphore *VulkanWindow::ImageAvailableSemaphore() const {
  return image_available_semaphores_[core_->CurrentFrame()].get();
}

uint32_t VulkanWindow::AcquireNextImage() {
  swap_chain_->AcquireNextImage(std::numeric_limits<uint64_t>::max(),
                                image_available_semaphores_[core_->CurrentFrame()]->Handle(), VK_NULL_HANDLE,
                                &image_index_);
  return image_index_;
}

void VulkanWindow::Rebuild() {
  core_->WaitGPU();
  swap_chain_.reset();
  core_->Device()->CreateSwapchain(
      surface_.get(), enable_hdr_ ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R8G8B8A8_UNORM,
      enable_hdr_ ? VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT : VK_COLOR_SPACE_SRGB_NONLINEAR_KHR, &swap_chain_);
}

void VulkanWindow::Present() {
  VkSemaphore render_finish_semaphore = render_finish_semaphores_[core_->CurrentFrame()]->Handle();

  VkSwapchainKHR swap_chain = swap_chain_->Handle();

  VkPresentInfoKHR presentInfo{};
  presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  presentInfo.waitSemaphoreCount = 1;
  presentInfo.pWaitSemaphores = &render_finish_semaphore;
  presentInfo.swapchainCount = 1;
  presentInfo.pSwapchains = &swap_chain;
  presentInfo.pImageIndices = &image_index_;

  auto result = vkQueuePresentKHR(present_queue_, &presentInfo);
  if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
    Rebuild();
  } else if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to present swap chain image");
  }
}

}  // namespace grassland::graphics::backend
