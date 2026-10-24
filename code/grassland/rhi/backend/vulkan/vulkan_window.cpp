// Include Vulkan before GLFW declares its surface functions.
#define GLFW_INCLUDE_VULKAN
#include "grassland/rhi/backend/vulkan/vulkan_window.h"

#include <cstdlib>
#include <optional>

#include "grassland/rhi/backend/vulkan/vulkan_swap_chain_image.h"
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

namespace grassland::rhi::backend::vulkan {

namespace {
std::optional<WindowPresentationOption> PresentationFormat(VkSurfaceFormatKHR native) {
  if (native.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
    if (native.format == VK_FORMAT_R8G8B8A8_UNORM)
      return WindowPresentationOption{PRESENT_MODE_VSYNC, IMAGE_FORMAT_RGBA8_UNORM};
    if (native.format == VK_FORMAT_B8G8R8A8_UNORM || native.format == VK_FORMAT_UNDEFINED)
      return WindowPresentationOption{PRESENT_MODE_VSYNC, IMAGE_FORMAT_BGRA8_UNORM};
  }
  if (native.colorSpace == VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT && native.format == VK_FORMAT_R16G16B16A16_SFLOAT)
    return WindowPresentationOption{PRESENT_MODE_VSYNC, IMAGE_FORMAT_RGBA16_FLOAT, COLOR_SPACE_EXTENDED_SRGB_LINEAR};
  if (native.colorSpace == VK_COLOR_SPACE_BT709_LINEAR_EXT && native.format == VK_FORMAT_R16G16B16A16_SFLOAT)
    return WindowPresentationOption{PRESENT_MODE_VSYNC, IMAGE_FORMAT_RGBA16_FLOAT, COLOR_SPACE_BT709_LINEAR};
  if (native.colorSpace == VK_COLOR_SPACE_HDR10_ST2084_EXT) {
    if (native.format == VK_FORMAT_A2B10G10R10_UNORM_PACK32)
      return WindowPresentationOption{PRESENT_MODE_VSYNC, IMAGE_FORMAT_RGB10A2_UNORM, COLOR_SPACE_HDR10_ST2084};
    if (native.format == VK_FORMAT_A2R10G10B10_UNORM_PACK32)
      return WindowPresentationOption{PRESENT_MODE_VSYNC, IMAGE_FORMAT_BGR10A2_UNORM, COLOR_SPACE_HDR10_ST2084};
  }
  return std::nullopt;
}
}  // namespace

VulkanWindow::VulkanWindow(VulkanDevice *device) : device_(device) {
}

VulkanWindow::~VulkanWindow() {
  glfw::UnregisterWindow(GLFWWindow());
  std::lock_guard<std::mutex> lock(device_->QueueMutex());
  vkDeviceWaitIdle(device_->NativeDevice());
  ReleaseNextImage();
  DestroySwapchain();
  if (surface_)
    vkDestroySurfaceKHR(device_->Instance(), surface_, nullptr);
}

VulkanWindow *VulkanWindow::Create(VulkanDevice *device, const WindowDesc &desc) {
  auto object = std::unique_ptr<VulkanWindow>(new VulkanWindow(device));
  if (!object->device_->SupportsSwapchain()) {
    ReportError(-1, "VK_KHR_swapchain is unavailable.");
    return nullptr;
  }
  if (!object->InitializeWindow(device, desc))
    return nullptr;
  if (!Check(glfwCreateWindowSurface(object->device_->Instance(), object->GLFWWindow(), nullptr, &object->surface_),
             "glfwCreateWindowSurface"))
    return nullptr;
  VkBool32 supported{};
  if (!Check(vkGetPhysicalDeviceSurfaceSupportKHR(object->device_->NativePhysicalDevice(),
                                                  object->device_->QueueFamily(), object->surface_, &supported),
             "Surface queue support") ||
      !supported) {
    ReportError(-1, "The graphics queue cannot present to this surface.");
    return nullptr;
  }
  if (!object->RebuildSwapChain())
    return nullptr;
  return object.release();
}

const void *VulkanWindow::DeviceIdentity() const {
  return device_;
}

ObjectPtr<Image> VulkanWindow::AcquireNativeFrame() {
  if (glfwGetWindowAttrib(GLFWWindow(), GLFW_ICONIFIED))
    return {};
  int width, height;
  glfwGetFramebufferSize(GLFWWindow(), &width, &height);
  if (width <= 0 || height <= 0)
    return {};
  if (invalid_)
    return {};
  auto &sync = acquire_sync_[current_sync_];
  const auto native = device_->NativeDevice();
  if (sync.submission_value) {
    const auto status = device_->Wait(sync.submission_value);
    if (!Check(status, "Acquire frame synchronization"))
      return {};
    sync.submission_value = 0;
  }
  if (sync.acquired) {
    auto status = vkWaitForFences(native, 1, &sync.fence, VK_TRUE, UINT64_MAX);
    if (!Check(status, "Acquire frame synchronization"))
      return {};
    status = vkResetFences(native, 1, &sync.fence);
    if (!Check(status, "Acquire frame synchronization"))
      return {};
    sync.acquired = false;
  }
  const auto semaphore = sync.semaphore;
  const auto fence = sync.fence;
  const auto result =
      vkAcquireNextImageKHR(device_->NativeDevice(), swapchain_, UINT64_MAX, semaphore, fence, &current_image_);
  if (result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR) {
    sync.acquired = true;
    const auto image = images_[current_image_];
    if (!initialized_images_[current_image_]) {
      // Establish the fixed default layout before exposing this image to recording.
      // This runs once per physical swapchain image, not once per CommandContext.
      const auto device = device_->NativeDevice();
      auto status = vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
      if (!Check(status, "Acquire frame synchronization"))
        return {};
      std::lock_guard<std::mutex> lock(device_->QueueMutex());
      VkCommandPool pool{};
      VkCommandBuffer commands{};
      status = [&]() -> VkResult {
        VkCommandPoolCreateInfo create_pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        create_pool.queueFamilyIndex = device_->QueueFamily();
        auto status = vkCreateCommandPool(device, &create_pool, nullptr, &pool);
        if (status != VK_SUCCESS)
          return status;
        VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocate.commandPool = pool;
        allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocate.commandBufferCount = 1;
        status = vkAllocateCommandBuffers(device, &allocate, &commands);
        if (status != VK_SUCCESS)
          return status;
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        status = vkBeginCommandBuffer(commands, &begin);
        if (status != VK_SUCCESS)
          return status;
        VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0,
                             nullptr, 0, nullptr, 1, &barrier);
        status = vkEndCommandBuffer(commands);
        if (status != VK_SUCCESS)
          return status;
        const auto semaphore = device_->Timeline();
        const auto value = device_->NextSerial();
        VkTimelineSemaphoreSubmitInfo timeline{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
        timeline.signalSemaphoreValueCount = 1;
        timeline.pSignalSemaphoreValues = &value;
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.pNext = &timeline;
        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores = &semaphore;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &commands;
        status = vkQueueSubmit(device_->NativeQueue(), 1, &submit, VK_NULL_HANDLE);
        if (status != VK_SUCCESS)
          return status;
        // Submitted commands must finish before their temporary pool is destroyed.
        if (!Check(device_->Wait(value), "Wait for image initialization"))
          std::abort();
        return VK_SUCCESS;
      }();
      if (pool)
        vkDestroyCommandPool(device, pool, nullptr);
      if (!Check(status, "Initialize swapchain image"))
        return {};
      initialized_images_[current_image_] = true;
    }
  } else if (result == VK_ERROR_OUT_OF_DATE_KHR) {
    invalid_ = true;
    return {};
  }
  if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
    Check(result, "vkAcquireNextImageKHR");
    return {};
  }
  return ObjectPtr<Image>::Adopt(new VulkanSwapChainImage(desc_, this, images_[current_image_],
                                                          image_views_[current_image_].Attachment(),
                                                          image_views_[current_image_].Binding()));
}

VkSemaphore VulkanWindow::AcquireSemaphore() const {
  const auto &sync = acquire_sync_[current_sync_];
  return sync.submission_value ? VK_NULL_HANDLE : sync.semaphore;
}

void VulkanWindow::MarkAcquireWaitSubmitted(uint64_t value) {
  acquire_sync_[current_sync_].submission_value = value;
}

VkSemaphore VulkanWindow::RenderSemaphore() const {
  return rendered_[current_image_];
}

VkResult VulkanWindow::PresentFrame() {
  const auto rendered = RenderSemaphore();
  VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
  present.waitSemaphoreCount = 1;
  present.pWaitSemaphores = &rendered;
  present.swapchainCount = 1;
  present.pSwapchains = &swapchain_;
  present.pImageIndices = &current_image_;
  const auto result = vkQueuePresentKHR(device_->NativeQueue(), &present);
  // The prepared frame has been presented; this slot is reused only after
  // its acquire fence and consuming submission have completed.
  current_sync_ = (current_sync_ + 1) % static_cast<uint32_t>(acquire_sync_.size());
  if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
    invalid_ = true;
  return result;
}

std::vector<WindowPresentationOption> VulkanWindow::GetSupportedPresentationOptions() const {
  const auto physical = device_->NativePhysicalDevice();
  VkSurfaceCapabilitiesKHR caps{};
  if (!Check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical, surface_, &caps), "Surface capabilities"))
    return {};
  if ((caps.supportedUsageFlags & (VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)) !=
      (VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)) {
    ReportError(-1, "Surface does not support color attachments and transfer clears.");
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
    const auto option = PresentationFormat(format);
    if (!option)
      continue;
    if (option->color_space != COLOR_SPACE_SRGB_NONLINEAR && !(caps.supportedUsageFlags & VK_IMAGE_USAGE_SAMPLED_BIT))
      continue;
    if (std::any_of(options.begin(), options.end(), [&](const auto &other) {
          return other.format == option->format && other.color_space == option->color_space;
        }))
      continue;
    options.push_back(*option);
    if (continuous)
      options.push_back({PRESENT_MODE_CONTINUOUS, option->format, option->color_space});
  }
  return options;
}

PresentMode VulkanWindow::GetPresentMode() const {
  return mode_;
}

void VulkanWindow::DestroySwapchain() {
  for (auto image : images_)
    device_->RenderTargets().Retire(image);
  image_views_.clear();
  // Device idle covers submitted waits; acquisition itself is a presentation-engine operation.
  for (auto &sync : acquire_sync_) {
    if (sync.acquired)
      vkWaitForFences(device_->NativeDevice(), 1, &sync.fence, VK_TRUE, UINT64_MAX);
    sync.submission_value = 0;
    if (sync.semaphore)
      vkDestroySemaphore(device_->NativeDevice(), sync.semaphore, nullptr);
    if (sync.fence)
      vkDestroyFence(device_->NativeDevice(), sync.fence, nullptr);
  }
  acquire_sync_.clear();
  current_sync_ = 0;
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

bool VulkanWindow::RebuildNativeSwapChain(PresentMode requested_mode) {
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
  if ((caps.supportedUsageFlags & (VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)) !=
      (VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)) {
    ReportError(-1, "Surface does not support color attachments and transfer clears.");
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
  auto format = std::find_if(formats.begin(), formats.end(), [&](const auto &native) {
    const auto option = PresentationFormat(native);
    return option && option->color_space == RequestedColorSpace();
  });
  if (format == formats.end()) {
    ReportError(-1, "No surface format for the requested color space.");
    return false;
  }
  const auto image_format = PresentationFormat(*format)->format;
  VkPresentModeKHR mode = VK_PRESENT_MODE_FIFO_KHR;
  if (requested_mode == PRESENT_MODE_CONTINUOUS) {
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
  create.imageFormat = NativeImageFormat(image_format);
  create.imageColorSpace = format->colorSpace;
  create.imageExtent = caps.currentExtent;
  if (create.imageExtent.width == UINT32_MAX) {
    create.imageExtent.width =
        std::clamp(static_cast<uint32_t>(width), caps.minImageExtent.width, caps.maxImageExtent.width);
    create.imageExtent.height =
        std::clamp(static_cast<uint32_t>(height), caps.minImageExtent.height, caps.maxImageExtent.height);
  }
  create.imageArrayLayers = 1;
  create.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
  const bool sampled = RequestedColorSpace() != COLOR_SPACE_SRGB_NONLINEAR;
  if (sampled) {
    if (!(caps.supportedUsageFlags & VK_IMAGE_USAGE_SAMPLED_BIT))
      return false;
    create.imageUsage |= VK_IMAGE_USAGE_SAMPLED_BIT;
  }
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
  DestroySwapchain();
  if (!Check(vkCreateSwapchainKHR(device, &create, nullptr, &swapchain_), "vkCreateSwapchainKHR"))
    return false;
  if (!EnumerateList([&](uint32_t *n, VkImage *p) { return vkGetSwapchainImagesKHR(device, swapchain_, n, p); },
                     images_))
    return false;
  image_views_.resize(images_.size());
  for (size_t i = 0; i < images_.size(); ++i) {
    if (!image_views_[i].Initialize(device, images_[i], image_format, sampled))
      return false;
  }
  acquire_sync_.resize(images_.size() + 1);
  for (auto &sync : acquire_sync_) {
    VkSemaphoreCreateInfo semaphore{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    if (!Check(vkCreateSemaphore(device, &semaphore, nullptr, &sync.semaphore), "Create acquire semaphore"))
      return false;
    VkFenceCreateInfo fence{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (!Check(vkCreateFence(device, &fence, nullptr, &sync.fence), "Create acquire fence"))
      return false;
  }
  initialized_images_.assign(images_.size(), false);
  rendered_.resize(images_.size());
  VkSemaphoreCreateInfo semaphore{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  for (auto &entry : rendered_)
    if (!Check(vkCreateSemaphore(device, &semaphore, nullptr, &entry), "Create present semaphore"))
      return false;
  desc_ = {create.imageExtent.width, create.imageExtent.height, image_format};
  mode_ = mode == VK_PRESENT_MODE_FIFO_KHR ? PRESENT_MODE_VSYNC : PRESENT_MODE_CONTINUOUS;
  return true;
}

}  // namespace grassland::rhi::backend::vulkan
