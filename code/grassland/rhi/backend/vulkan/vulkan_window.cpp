#include "grassland/rhi/backend/vulkan/vulkan_window.h"

#include "grassland/rhi/backend/vulkan/vulkan_swap_chain_image.h"
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

namespace grassland::rhi::backend::vulkan {

VulkanWindow::VulkanWindow(VulkanDevice *device) : device_(device) {
}

VulkanWindow::~VulkanWindow() {
  std::lock_guard<std::mutex> lock(device_->QueueMutex());
  vkDeviceWaitIdle(device_->NativeDevice());
  DestroySwapchain();
  if (surface_)
    vkDestroySurfaceKHR(device_->Instance(), surface_, nullptr);
}

bool VulkanWindow::Initialize(const WindowDesc &desc) {
  if (!device_->SupportsSwapchain()) {
    ReportError(-1, "VK_KHR_swapchain is unavailable.");
    return false;
  }
  if (!InitializeWindow(desc) ||
      !Check(glfwCreateWindowSurface(device_->Instance(), NativeWindow(), nullptr, &surface_),
             "glfwCreateWindowSurface"))
    return false;
  VkBool32 supported{};
  if (!Check(vkGetPhysicalDeviceSurfaceSupportKHR(device_->NativePhysicalDevice(), device_->QueueFamily(), surface_,
                                                  &supported),
             "Surface queue support") ||
      !supported) {
    ReportError(-1, "The graphics queue cannot present to this surface.");
    return false;
  }
  requested_ = desc.present_mode;
  return Rebuild();
}

bool VulkanWindow::IsFrameValid(uint64_t generation) const {
  return generation_ == generation && valid_;
}

const void *VulkanWindow::DeviceIdentity() const {
  return device_;
}

void VulkanWindow::CancelFrame(uint64_t generation) {
  if (generation_ == generation) {
    acquired_ = false;
    valid_ = false;
    pending_acquire_ = VK_NULL_HANDLE;
  }
}

VkResult VulkanWindow::AcquireNative(VkSemaphore semaphore,
                                     VkFence fence,
                                     uint32_t &index,
                                     VkImage &image,
                                     VkSemaphore &rendered) {
  const auto result = vkAcquireNextImageKHR(device_->NativeDevice(), swapchain_, UINT64_MAX, semaphore, fence, &index);
  if (result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR) {
    image = images_[index];
    rendered = rendered_[index];
    acquired_ = true;
    pending_acquire_ = fence;
  } else if (result == VK_ERROR_OUT_OF_DATE_KHR)
    valid_ = false;
  return result;
}

VkResult VulkanWindow::PresentFrame(uint64_t generation, uint32_t index, VkSemaphore rendered) {
  if (!IsFrameValid(generation))
    return VK_ERROR_OUT_OF_DATE_KHR;
  VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
  present.waitSemaphoreCount = 1;
  present.pWaitSemaphores = &rendered;
  present.swapchainCount = 1;
  present.pSwapchains = &swapchain_;
  present.pImageIndices = &index;
  const auto result = vkQueuePresentKHR(device_->NativeQueue(), &present);
  acquired_ = false;
  pending_acquire_ = VK_NULL_HANDLE;
  if (result == VK_SUBOPTIMAL_KHR || result == VK_ERROR_OUT_OF_DATE_KHR) {
    valid_ = false;
    return VK_SUCCESS;
  }
  if (result != VK_SUCCESS)
    valid_ = false;
  return result;
}

ObjectPtr<Image> VulkanWindow::AcquireNativeFrame() {
  if (glfwGetWindowAttrib(NativeWindow(), GLFW_ICONIFIED))
    return {};
  int width, height;
  glfwGetFramebufferSize(NativeWindow(), &width, &height);
  if (width <= 0 || height <= 0)
    return {};
  if (!valid_ || width != width_ || height != height_) {
    if (!Rebuild())
      return {};
  }
  if (acquired_) {
    ReportError(-1, "Present or release the previous acquired frame first.");
    return {};
  }
  auto frame = ObjectPtr<VulkanSwapChainImage>::Adopt(
      new VulkanSwapChainImage(desc_, this, generation_, device_->NativeDevice()));
  return frame->Acquire() ? ObjectPtr<Image>(frame) : ObjectPtr<Image>{};
}

bool VulkanWindow::SetNativePresentMode(PresentMode mode) {
  if (requested_ == mode && valid_)
    return true;
  requested_ = mode;
  valid_ = false;
  return Rebuild();
}

std::vector<WindowPresentationOption> VulkanWindow::GetSupportedPresentationOptions() const {
  const auto physical = device_->NativePhysicalDevice();
  VkSurfaceCapabilitiesKHR caps{};
  if (!Check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical, surface_, &caps), "Surface capabilities"))
    return {};
  if (!(caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT)) {
    ReportError(-1, "Surface does not support transfer clears.");
    return {};
  }
  std::vector<VkSurfaceFormatKHR> formats;
  std::vector<VkPresentModeKHR> modes;
  if (!EnumerateList(
          [&](uint32_t *n, VkSurfaceFormatKHR *p) {
            return vkGetPhysicalDeviceSurfaceFormatsKHR(physical, surface_, n, p);
          },
          formats) ||
      !EnumerateList(
          [&](uint32_t *n, VkPresentModeKHR *p) {
            return vkGetPhysicalDeviceSurfacePresentModesKHR(physical, surface_, n, p);
          },
          modes))
    return {};
  const bool continuous = std::find(modes.begin(), modes.end(), VK_PRESENT_MODE_IMMEDIATE_KHR) != modes.end() ||
                          std::find(modes.begin(), modes.end(), VK_PRESENT_MODE_MAILBOX_KHR) != modes.end();
  std::vector<WindowPresentationOption> options;
  for (const auto &format : formats) {
    // Match the formats and color space implemented by Rebuild.
    if (format.colorSpace != VK_COLOR_SPACE_SRGB_NONLINEAR_KHR ||
        (format.format != VK_FORMAT_R8G8B8A8_UNORM && format.format != VK_FORMAT_B8G8R8A8_UNORM))
      continue;
    const auto image_format =
        format.format == VK_FORMAT_R8G8B8A8_UNORM ? ImageFormat::RGBA8Unorm : ImageFormat::BGRA8Unorm;
    if (std::any_of(options.begin(), options.end(), [&](const auto &option) { return option.format == image_format; }))
      continue;
    options.push_back({PresentMode::VSync, image_format});
    if (continuous)
      options.push_back({PresentMode::Continuous, image_format});
  }
  return options;
}

PresentMode VulkanWindow::GetPresentMode() const {
  return mode_;
}

void VulkanWindow::DestroySwapchain() {
  for (auto semaphore : rendered_)
    if (semaphore)
      vkDestroySemaphore(device_->NativeDevice(), semaphore, nullptr);
  rendered_.clear();
  images_.clear();
  if (swapchain_)
    vkDestroySwapchainKHR(device_->NativeDevice(), swapchain_, nullptr);
  swapchain_ = VK_NULL_HANDLE;
}

bool VulkanWindow::Rebuild() {
  if (glfwGetWindowAttrib(NativeWindow(), GLFW_ICONIFIED))
    return true;
  int width, height;
  glfwGetFramebufferSize(NativeWindow(), &width, &height);
  if (width <= 0 || height <= 0)
    return true;
  const auto physical = device_->NativePhysicalDevice();
  const auto device = device_->NativeDevice();
  VkSurfaceCapabilitiesKHR caps{};
  if (!Check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical, surface_, &caps), "Surface capabilities"))
    return false;
  if (!(caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT)) {
    ReportError(-1, "Surface does not support transfer clears.");
    return false;
  }
  std::vector<VkSurfaceFormatKHR> formats;
  std::vector<VkPresentModeKHR> modes;
  if (!EnumerateList(
          [&](uint32_t *n, VkSurfaceFormatKHR *p) {
            return vkGetPhysicalDeviceSurfaceFormatsKHR(physical, surface_, n, p);
          },
          formats) ||
      !EnumerateList(
          [&](uint32_t *n, VkPresentModeKHR *p) {
            return vkGetPhysicalDeviceSurfacePresentModesKHR(physical, surface_, n, p);
          },
          modes))
    return false;
  auto format = std::find_if(formats.begin(), formats.end(), [](const auto &f) {
    return (f.format == VK_FORMAT_B8G8R8A8_UNORM || f.format == VK_FORMAT_R8G8B8A8_UNORM) &&
           f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
  });
  if (format == formats.end()) {
    ReportError(-1, "No supported UNORM swapchain format.");
    return false;
  }
  VkPresentModeKHR mode = VK_PRESENT_MODE_FIFO_KHR;
  if (requested_ == PresentMode::Continuous) {
    for (auto candidate : {VK_PRESENT_MODE_IMMEDIATE_KHR, VK_PRESENT_MODE_MAILBOX_KHR}) {
      if (std::find(modes.begin(), modes.end(), candidate) != modes.end()) {
        mode = candidate;
        break;
      }
    }
  }
  VkSwapchainCreateInfoKHR create{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
  create.surface = surface_;
  create.minImageCount = std::max(3u, caps.minImageCount);
  if (caps.maxImageCount)
    create.minImageCount = std::min(create.minImageCount, caps.maxImageCount);
  create.imageFormat = format->format;
  create.imageColorSpace = format->colorSpace;
  create.imageExtent = caps.currentExtent;
  if (create.imageExtent.width == UINT32_MAX) {
    create.imageExtent.width =
        std::clamp(static_cast<uint32_t>(width), caps.minImageExtent.width, caps.maxImageExtent.width);
    create.imageExtent.height =
        std::clamp(static_cast<uint32_t>(height), caps.minImageExtent.height, caps.maxImageExtent.height);
  }
  create.imageArrayLayers = 1;
  create.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  create.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  create.preTransform = caps.currentTransform;
  for (auto alpha : {VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR, VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
                     VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR, VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR}) {
    if (caps.supportedCompositeAlpha & alpha) {
      create.compositeAlpha = alpha;
      break;
    }
  }
  create.presentMode = mode;
  create.clipped = VK_TRUE;
  std::lock_guard<std::mutex> lock(device_->QueueMutex());
  if (swapchain_) {
    if (pending_acquire_) {
      VkResult result;
      do {
        result = vkWaitForFences(device, 1, &pending_acquire_, VK_TRUE, 50000000);
      } while (result == VK_TIMEOUT);
      if (!Check(result, "Wait for pending image acquisition"))
        return false;
    }
    if (!Check(vkDeviceWaitIdle(device), "Wait before swapchain replacement"))
      return false;
  }
  ++generation_;
  valid_ = false;
  acquired_ = false;
  pending_acquire_ = VK_NULL_HANDLE;
  DestroySwapchain();
  if (!Check(vkCreateSwapchainKHR(device, &create, nullptr, &swapchain_), "vkCreateSwapchainKHR"))
    return false;
  if (!EnumerateList([&](uint32_t *n, VkImage *p) { return vkGetSwapchainImagesKHR(device, swapchain_, n, p); },
                     images_))
    return false;
  rendered_.resize(images_.size());
  VkSemaphoreCreateInfo semaphore{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  for (auto &entry : rendered_)
    if (!Check(vkCreateSemaphore(device, &semaphore, nullptr, &entry), "Create present semaphore"))
      return false;
  desc_ = {create.imageExtent.width, create.imageExtent.height,
           format->format == VK_FORMAT_B8G8R8A8_UNORM ? ImageFormat::BGRA8Unorm : ImageFormat::RGBA8Unorm};
  mode_ = mode == VK_PRESENT_MODE_FIFO_KHR ? PresentMode::VSync : PresentMode::Continuous;
  width_ = width;
  height_ = height;
  valid_ = true;
  return true;
}

}  // namespace grassland::rhi::backend::vulkan
