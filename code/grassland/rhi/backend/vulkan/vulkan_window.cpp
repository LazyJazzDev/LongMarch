// Include Vulkan before GLFW declares its surface functions.
#define GLFW_INCLUDE_VULKAN
#include "grassland/rhi/backend/vulkan/vulkan_window.h"

#include "grassland/rhi/backend/vulkan/vulkan_swap_chain_image.h"
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

namespace grassland::rhi::backend::vulkan {

VulkanWindow::VulkanWindow(VulkanDevice *device) : device_(device) {
}

VulkanWindow::~VulkanWindow() {
  glfw::UnregisterWindow(GLFWWindow());
  std::lock_guard<std::mutex> lock(device_->QueueMutex());
  vkDeviceWaitIdle(device_->NativeDevice());
  ReleaseNextImage();
  DestroySwapchain();
  if (initialization_fence_)
    vkDestroyFence(device_->NativeDevice(), initialization_fence_, nullptr);
  if (initialization_pool_)
    vkDestroyCommandPool(device_->NativeDevice(), initialization_pool_, nullptr);
  if (surface_)
    vkDestroySurfaceKHR(device_->Instance(), surface_, nullptr);
}

bool VulkanWindow::Initialize(const WindowDesc &desc) {
  if (!device_->SupportsSwapchain()) {
    ReportError(-1, "VK_KHR_swapchain is unavailable.");
    return false;
  }
  if (!InitializeWindow(desc) ||
      !Check(glfwCreateWindowSurface(device_->Instance(), GLFWWindow(), nullptr, &surface_), "glfwCreateWindowSurface"))
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
  return RebuildSwapChain();
}

bool VulkanWindow::IsFrameValid(uint64_t generation) const {
  return generation_ == generation && valid_;
}

const void *VulkanWindow::DeviceIdentity() const {
  return device_;
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
    if (!initialized_images_[index]) {
      // Establish the fixed default layout before exposing this image to recording.
      // This runs once per physical swapchain image, not once per CommandContext.
      const auto device = device_->NativeDevice();
      auto status = vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
      if (status != VK_SUCCESS)
        return status;
      std::lock_guard<std::mutex> lock(device_->QueueMutex());
      if (!initialization_pool_) {
        VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pool.queueFamilyIndex = device_->QueueFamily();
        status = vkCreateCommandPool(device, &pool, nullptr, &initialization_pool_);
        if (status != VK_SUCCESS)
          return status;
      }
      if (!initialization_commands_) {
        VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocate.commandPool = initialization_pool_;
        allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocate.commandBufferCount = 1;
        status = vkAllocateCommandBuffers(device, &allocate, &initialization_commands_);
        if (status != VK_SUCCESS)
          return status;
      }
      if (!initialization_fence_) {
        VkFenceCreateInfo create{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        status = vkCreateFence(device, &create, nullptr, &initialization_fence_);
        if (status != VK_SUCCESS)
          return status;
      }
      status = vkResetCommandPool(device, initialization_pool_, 0);
      if (status != VK_SUCCESS)
        return status;
      status = vkResetFences(device, 1, &initialization_fence_);
      if (status != VK_SUCCESS)
        return status;
      VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      status = vkBeginCommandBuffer(initialization_commands_, &begin);
      if (status != VK_SUCCESS)
        return status;
      VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
      barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
      barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.image = image;
      barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
      vkCmdPipelineBarrier(initialization_commands_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                           VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
      status = vkEndCommandBuffer(initialization_commands_);
      if (status != VK_SUCCESS)
        return status;
      VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
      submit.commandBufferCount = 1;
      submit.pCommandBuffers = &initialization_commands_;
      status = vkQueueSubmit(device_->NativeQueue(), 1, &submit, initialization_fence_);
      if (status != VK_SUCCESS)
        return status;
      status = vkWaitForFences(device, 1, &initialization_fence_, VK_TRUE, UINT64_MAX);
      if (status != VK_SUCCESS)
        return status;
      initialized_images_[index] = true;
    }
  } else if (result == VK_ERROR_OUT_OF_DATE_KHR)
    valid_ = false;
  return result;
}

VkResult VulkanWindow::PresentFrame(uint64_t generation, uint32_t index, VkSemaphore rendered) {
#if !defined(NDEBUG)
  if (!IsFrameValid(generation))
    return VK_ERROR_OUT_OF_DATE_KHR;
#endif
  VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
  present.waitSemaphoreCount = 1;
  present.pWaitSemaphores = &rendered;
  present.swapchainCount = 1;
  present.pSwapchains = &swapchain_;
  present.pImageIndices = &index;
  const auto result = vkQueuePresentKHR(device_->NativeQueue(), &present);
  acquired_ = false;
  if (result == VK_SUBOPTIMAL_KHR || result == VK_ERROR_OUT_OF_DATE_KHR) {
    valid_ = false;
    return VK_SUCCESS;
  }
  if (result != VK_SUCCESS)
    valid_ = false;
  return result;
}

ObjectPtr<Image> VulkanWindow::AcquireNativeFrame() {
  if (glfwGetWindowAttrib(GLFWWindow(), GLFW_ICONIFIED))
    return {};
  int width, height;
  glfwGetFramebufferSize(GLFWWindow(), &width, &height);
  if (width <= 0 || height <= 0)
    return {};
#if !defined(NDEBUG)
  if (!valid_ || width != width_ || height != height_)
    return {};
#endif
#if !defined(NDEBUG)
  if (acquired_) {
    ReportError(-1, "Present or release the previous acquired frame first.");
    return {};
  }
#endif
  auto frame = ObjectPtr<VulkanSwapChainImage>::Adopt(new VulkanSwapChainImage(desc_, this, generation_, device_));
  return frame->Acquire() ? ObjectPtr<Image>(frame) : ObjectPtr<Image>{};
}

bool VulkanWindow::SetNativePresentMode(PresentMode mode) {
  requested_ = mode;
  return true;
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
    // Match the formats and color space implemented by RebuildNativeSwapChain.
    if (format.colorSpace != VK_COLOR_SPACE_SRGB_NONLINEAR_KHR ||
        (format.format != VK_FORMAT_R8G8B8A8_UNORM && format.format != VK_FORMAT_B8G8R8A8_UNORM))
      continue;
    const auto image_format =
        format.format == VK_FORMAT_R8G8B8A8_UNORM ? IMAGE_FORMAT_RGBA8_UNORM : IMAGE_FORMAT_BGRA8_UNORM;
    if (std::any_of(options.begin(), options.end(), [&](const auto &option) { return option.format == image_format; }))
      continue;
    options.push_back({PRESENT_MODE_VSYNC, image_format});
    if (continuous)
      options.push_back({PRESENT_MODE_CONTINUOUS, image_format});
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
  initialized_images_.clear();
  if (swapchain_)
    vkDestroySwapchainKHR(device_->NativeDevice(), swapchain_, nullptr);
  swapchain_ = VK_NULL_HANDLE;
}

bool VulkanWindow::RebuildNativeSwapChain() {
  valid_ = false;
  if (glfwGetWindowAttrib(GLFWWindow(), GLFW_ICONIFIED))
    return true;
  int width, height;
  glfwGetFramebufferSize(GLFWWindow(), &width, &height);
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
  if (requested_ == PRESENT_MODE_CONTINUOUS) {
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
    if (!Check(vkDeviceWaitIdle(device), "Wait before swapchain replacement"))
      return false;
  }
  ++generation_;
  valid_ = false;
  acquired_ = false;
  DestroySwapchain();
  if (!Check(vkCreateSwapchainKHR(device, &create, nullptr, &swapchain_), "vkCreateSwapchainKHR"))
    return false;
  if (!EnumerateList([&](uint32_t *n, VkImage *p) { return vkGetSwapchainImagesKHR(device, swapchain_, n, p); },
                     images_))
    return false;
  initialized_images_.assign(images_.size(), false);
  rendered_.resize(images_.size());
  VkSemaphoreCreateInfo semaphore{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  for (auto &entry : rendered_)
    if (!Check(vkCreateSemaphore(device, &semaphore, nullptr, &entry), "Create present semaphore"))
      return false;
  desc_ = {create.imageExtent.width, create.imageExtent.height,
           format->format == VK_FORMAT_B8G8R8A8_UNORM ? IMAGE_FORMAT_BGRA8_UNORM : IMAGE_FORMAT_RGBA8_UNORM};
  mode_ = mode == VK_PRESENT_MODE_FIFO_KHR ? PRESENT_MODE_VSYNC : PRESENT_MODE_CONTINUOUS;
  width_ = width;
  height_ = height;
  valid_ = true;
  return true;
}

}  // namespace grassland::rhi::backend::vulkan
