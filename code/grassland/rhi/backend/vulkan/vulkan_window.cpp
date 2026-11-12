#define GLFW_INCLUDE_VULKAN
#include "grassland/rhi/backend/vulkan/vulkan_window_impl.h"
// Include Vulkan before GLFW declares its surface functions.
#include <cstdlib>
#include <optional>

#include "grassland/rhi/backend/vulkan/vulkan_queue.h"
#include "grassland/rhi/backend/vulkan/vulkan_swap_chain_image.h"
#include "grassland/rhi/backend/vulkan/vulkan_window.h"
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <list>
#include <mutex>
#include <thread>

#include "grassland/rhi/backend/backend_image_presenter.h"
#include "grassland/rhi/backend/backend_submission_token.h"
#include "grassland/rhi/backend/backend_window.h"
#include "grassland/rhi/backend/backend_window_imgui.h"
#include "grassland/rhi/rhi_device.h"
#include "grassland/rhi/rhi_queue.h"
#define GLFW_INCLUDE_NONE
#ifdef _WIN32
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#endif

namespace grassland::rhi::backend::vulkan {

namespace {
std::optional<WindowPresentationOption> PresentationFormat(VkSurfaceFormatKHR native) {
  if (native.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
    if (native.format == VK_FORMAT_R8G8B8A8_UNORM)
      return WindowPresentationOption{PRESENT_MODE_VSYNC, IMAGE_FORMAT_R8G8B8A8_UNORM};
    if (native.format == VK_FORMAT_B8G8R8A8_UNORM || native.format == VK_FORMAT_UNDEFINED)
      return WindowPresentationOption{PRESENT_MODE_VSYNC, IMAGE_FORMAT_B8G8R8A8_UNORM};
  }
  if (native.colorSpace == VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT && native.format == VK_FORMAT_R16G16B16A16_SFLOAT)
    return WindowPresentationOption{PRESENT_MODE_VSYNC, IMAGE_FORMAT_R16G16B16A16_SFLOAT,
                                    COLOR_SPACE_EXTENDED_SRGB_LINEAR};
  if (native.colorSpace == VK_COLOR_SPACE_BT709_LINEAR_EXT && native.format == VK_FORMAT_R16G16B16A16_SFLOAT)
    return WindowPresentationOption{PRESENT_MODE_VSYNC, IMAGE_FORMAT_R16G16B16A16_SFLOAT, COLOR_SPACE_BT709_LINEAR};
  if (native.colorSpace == VK_COLOR_SPACE_HDR10_ST2084_EXT) {
    if (native.format == VK_FORMAT_A2B10G10R10_UNORM_PACK32)
      return WindowPresentationOption{PRESENT_MODE_VSYNC, IMAGE_FORMAT_R10G10B10A2_UNORM, COLOR_SPACE_HDR10_ST2084};
    if (native.format == VK_FORMAT_A2R10G10B10_UNORM_PACK32)
      return WindowPresentationOption{PRESENT_MODE_VSYNC, IMAGE_FORMAT_B10G10R10A2_UNORM, COLOR_SPACE_HDR10_ST2084};
  }
  return std::nullopt;
}
}  // namespace

VulkanWindowImpl::VulkanWindowImpl(IVulkanDevice *device) : device_(device) {
}

VulkanWindowImpl::~VulkanWindowImpl() {
  auto queue = queue_;
  glfw::UnregisterWindow(GLFWWindow());
  std::lock_guard<std::mutex> lock(queue->QueueMutex());
  vkQueueWaitIdle(queue->NativeQueue());
  ReleaseNextImage();
  DestroySwapchain();
  if (surface_)
    vkDestroySurfaceKHR(device_->Instance(), surface_, nullptr);

  DestroyWindow();
}

rhi::IWindow *VulkanWindowImpl::Create(IVulkanDevice *device, const WindowDesc &desc) {
  auto object = std::unique_ptr<VulkanWindowImpl>(new VulkanWindowImpl(device));
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

ObjectPtr<IVulkanImage> VulkanWindowImpl::AcquireNativeFrame() {
  auto queue = queue_;
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
    const auto status = queue->Wait(sync.submission_value);
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
    {
      std::lock_guard<std::mutex> lock(queue->QueueMutex());
      const VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
      const uint64_t binary_value = 0;
      const auto value = queue->NextSerial();
      const auto completed = queue->GetTimeline()->NativeSemaphore();
      VkTimelineSemaphoreSubmitInfo timeline{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
      timeline.waitSemaphoreValueCount = timeline.signalSemaphoreValueCount = 1;
      timeline.pWaitSemaphoreValues = &binary_value;
      timeline.pSignalSemaphoreValues = &value;
      VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
      submit.pNext = &timeline;
      submit.waitSemaphoreCount = submit.signalSemaphoreCount = 1;
      submit.pWaitSemaphores = &semaphore;
      submit.pWaitDstStageMask = &stage;
      submit.pSignalSemaphores = &completed;
      if (!Check(vkQueueSubmit(queue->NativeQueue(), 1, &submit, VK_NULL_HANDLE), "Consume swapchain acquisition"))
        return {};
      sync.submission_value = value;
    }
    const auto image = images_[current_image_];
    if (!initialized_images_[current_image_]) {
      // Establish the fixed default layout before exposing this image to recording.
      // This runs once per physical swapchain image, not once per CommandContext.
      const auto device = device_->NativeDevice();
      auto status = vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
      if (!Check(status, "Acquire frame synchronization"))
        return {};
      std::lock_guard<std::mutex> lock(queue->QueueMutex());
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
        const auto semaphore = queue->GetTimeline()->NativeSemaphore();
        const auto value = queue->NextSerial();
        VkTimelineSemaphoreSubmitInfo timeline{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
        timeline.signalSemaphoreValueCount = 1;
        timeline.pSignalSemaphoreValues = &value;
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.pNext = &timeline;
        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores = &semaphore;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &commands;
        status = vkQueueSubmit(queue->NativeQueue(), 1, &submit, VK_NULL_HANDLE);
        if (status != VK_SUCCESS)
          return status;
        // Submitted commands must finish before their temporary pool is destroyed.
        if (!Check(queue->Wait(value), "Wait for image initialization"))
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
  return ObjectPtr<IVulkanImage>::Adopt(
      IVulkanSwapChainImage::Create(desc_, this, images_[current_image_], image_views_[current_image_].Attachment()));
}

VkSemaphore VulkanWindowImpl::RenderSemaphore() const {
  return rendered_[current_image_];
}

VkResult VulkanWindowImpl::PresentFrame() {
  auto queue = queue_;
  const auto rendered = RenderSemaphore();
  VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
  present.waitSemaphoreCount = 1;
  present.pWaitSemaphores = &rendered;
  present.swapchainCount = 1;
  present.pSwapchains = &swapchain_;
  present.pImageIndices = &current_image_;
  const auto result = vkQueuePresentKHR(queue->NativeQueue(), &present);
  // The prepared frame has been presented; this slot is reused only after
  // its acquire fence and consuming submission have completed.
  current_sync_ = (current_sync_ + 1) % static_cast<uint32_t>(acquire_sync_.size());
  if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
    invalid_ = true;
  return result;
}

std::vector<WindowPresentationOption> VulkanWindowImpl::GetSupportedPresentationOptions() const {
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

PresentMode VulkanWindowImpl::GetPresentMode() const {
  return mode_;
}

void VulkanWindowImpl::DestroySwapchain() {
  for (auto image : images_)
    device_->RenderTargets().Retire(image);
  image_views_.clear();
  // Main queue idle covers submitted waits; acquisition itself is a presentation-engine operation.
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

bool VulkanWindowImpl::RebuildNativeSwapChain(PresentMode requested_mode) {
  auto queue = queue_;
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
  create.imageUsage =
      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
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
  std::lock_guard<std::mutex> lock(queue->QueueMutex());
  if (swapchain_) {
    if (!Check(vkQueueWaitIdle(queue->NativeQueue()), "Wait before swapchain replacement"))
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
    if (!image_views_[i].Initialize(device, images_[i], image_format, false))
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

rhi::IWindow *IVulkanWindow::Create(IVulkanDevice *device, const WindowDesc &desc) {
  return VulkanWindowImpl::Create(device, desc);
}

bool VulkanWindowImpl::EnableImGui(bool enabled, const char *font_file_path, float font_size) {
  if (!glfw::IsWindowThread() || processing_imgui_)
    return false;
  if (!enabled) {
    imgui_.reset();
    return true;
  }
  if (!imgui_)
    imgui_ =
        WindowImGui::Create(rhi::QueryInterfacePtr<rhi::IDevice>(device_).Get(), window_, font_file_path, font_size);
  return bool(imgui_);
}

ImGuiContext *VulkanWindowImpl::GetImGuiContext() const {
  return imgui_ ? imgui_->Context() : nullptr;
}

bool VulkanWindowImpl::UpdateImGui(const std::function<void()> &callback) {
  if (!imgui_)
    return true;
  if (!glfw::IsWindowThread() || processing_imgui_)
    return false;

  struct ProcessingScope {
    bool &flag;

    ProcessingScope(bool &value) : flag(value) {
      flag = true;
    }

    ~ProcessingScope() {
      flag = false;
    }
  } scope(processing_imgui_);

  return imgui_->Update(callback);
}

const void *VulkanWindowImpl::DeviceIdentity() const {
  return device_;
}

bool VulkanWindowImpl::RecordPresentImage(ICommandContext *commands, IImage *image, ColorSpace color_space) {
  ObjPtr<IImage> frame = GetNextImage();
  if (!frame)
    return false;
  if (!presenter_)
    presenter_ = std::make_unique<ImagePresenter>(rhi::QueryInterfacePtr<rhi::IDevice>(device_).Get());
  return presenter_->Record(commands, image, frame.Get(), color_space, GetColorSpace(), GetHDRReferenceWhiteScale());
}

bool VulkanWindowImpl::RecordImGui(rhi::ICommandContext *commands, rhi::IImage *image) {
  return RecordImGui(commands, image, GetColorSpace(), GetHDRReferenceWhiteScale());
}

bool VulkanWindowImpl::RecordImGui(rhi::ICommandContext *commands,
                                   rhi::IImage *image,
                                   ColorSpace color_space,
                                   float white_scale) {
  if (!imgui_)
    return true;
  if (!glfw::IsWindowThread() || processing_imgui_)
    return false;

  struct ProcessingScope {
    bool &flag;

    ProcessingScope(bool &value) : flag(value) {
      flag = true;
    }

    ~ProcessingScope() {
      flag = false;
    }
  } scope(processing_imgui_);

  return imgui_->Record(commands, image, color_space, white_scale);
}

rhi::IImage *VulkanWindowImpl::GetNextImage() {
#if !defined(NDEBUG)
  if (!glfw::IsWindowThread()) {
    ReportError(-1, "GetNextImage must run on the window thread.");
    std::abort();
  }
#endif
  if (invalid_)
    return nullptr;
  int width, height;
  glfwGetFramebufferSize(window_, &width, &height);
  if (glfwGetWindowAttrib(window_, GLFW_ICONIFIED) || width <= 0 || height <= 0)
    return nullptr;
  std::lock_guard<std::mutex> lock(next_image_mutex_);
#if !defined(NDEBUG)
  if (!next_image_) {
    ReportError(-1, "No prepared swapchain image for a drawable window.");
    std::abort();
  }
#endif
  return rhi::QueryInterfacePtr<IImage>(next_image_.Get()).Detach();
}

IVulkanQueue *VulkanWindowImpl::OwnerQueue() const {
  return queue_;
}

bool VulkanWindowImpl::UpdateNextImage() {
  auto image = AcquireNativeFrame();
  std::lock_guard<std::mutex> lock(next_image_mutex_);
  next_image_ = std::move(image);
  int width, height;
  glfwGetFramebufferSize(window_, &width, &height);
  if (glfwGetWindowAttrib(window_, GLFW_ICONIFIED) || width <= 0 || height <= 0)
    return true;
  // A native OUT_OF_DATE result defers recovery to an explicit rebuild.
  if (next_image_ || invalid_)
    return true;
  invalid_ = true;
  return false;
}

void VulkanWindowImpl::ReleaseNextImage() {
  std::lock_guard<std::mutex> lock(next_image_mutex_);
  next_image_.Reset();
}

bool VulkanWindowImpl::IsSwapChainInvalid() const {
  return invalid_;
}

bool VulkanWindowImpl::RebuildSwapChain() {
#if !defined(NDEBUG)
  if (!glfw::IsWindowThread())
    return false;
#endif
  const auto options = GetSupportedPresentationOptions();
  if (std::none_of(options.begin(), options.end(),
                   [&](const auto &option) { return option.color_space == requested_color_space_; })) {
    ReportError(-1, "Requested swapchain color space is unavailable.");
    return false;
  }
  invalid_ = true;
  ReleaseNextImage();
  if (!RebuildNativeSwapChain(requested_mode_))
    return false;
  int width, height;
  glfwGetFramebufferSize(window_, &width, &height);
  if (glfwGetWindowAttrib(window_, GLFW_ICONIFIED) || width <= 0 || height <= 0)
    return true;
  color_space_ = requested_color_space_;
  invalid_ = false;
  if (UpdateNextImage())
    return true;
  invalid_ = true;
  return false;
}

bool VulkanWindowImpl::SetPresentMode(PresentMode mode) {
#if !defined(NDEBUG)
  if (!glfw::IsWindowThread())
    return false;
#endif
  requested_mode_ = mode;
  return RebuildSwapChain();
}

bool VulkanWindowImpl::SetColorSpace(ColorSpace color_space) {
#if !defined(NDEBUG)
  if (!glfw::IsWindowThread())
    return false;
#endif
  const auto options = GetSupportedPresentationOptions();
  if (std::none_of(options.begin(), options.end(),
                   [&](const auto &option) { return option.color_space == color_space; }))
    return false;
  const auto previous = requested_color_space_;
  requested_color_space_ = color_space;
  if (RebuildSwapChain())
    return true;
  requested_color_space_ = previous;
  return false;
}

ColorSpace VulkanWindowImpl::GetColorSpace() const {
  return color_space_;
}

float VulkanWindowImpl::GetHDRReferenceWhiteScale() const {
#ifdef _WIN32
  float nits;
  return GetHDRReferenceWhiteNits(nits) ? nits / 80.0f : 1.0f;
#else
  // Linux uses fixed source encoding parameters; macOS uses relative EDR.
  return 1.0f;
#endif
}

float VulkanWindowImpl::GetHDRHeadroom() const {
#ifdef __APPLE__
  return CocoaHDRHeadroom(GLFWWindow());
#endif
#ifdef _WIN32
  float white;
  if (!GetHDRReferenceWhiteNits(white))
    return 10.0f;
  const auto monitor = MonitorFromWindow(glfwGetWin32Window(window_), MONITOR_DEFAULTTONEAREST);
  Microsoft::WRL::ComPtr<IDXGIFactory1> factory;
  if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))))
    return 10.0f;
  for (UINT a = 0;; ++a) {
    Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
    if (factory->EnumAdapters1(a, &adapter) != S_OK)
      break;
    for (UINT o = 0;; ++o) {
      Microsoft::WRL::ComPtr<IDXGIOutput> output;
      if (adapter->EnumOutputs(o, &output) != S_OK)
        break;
      Microsoft::WRL::ComPtr<IDXGIOutput6> output6;
      DXGI_OUTPUT_DESC1 desc{};
      if (FAILED(output.As(&output6)) || FAILED(output6->GetDesc1(&desc)) || desc.Monitor != monitor)
        continue;
      const float ratio = desc.MaxLuminance / white;
      return std::isfinite(ratio) && ratio > 0.0f ? std::max(1.0f, ratio) : 10.0f;
    }
  }
#endif
  return 10.0f;
}

bool VulkanWindowImpl::GetHDRReferenceWhiteNits(float &nits) const {
  if (!glfw::IsWindowThread())
    return false;
#ifdef _WIN32
  MONITORINFOEXW monitor{};
  monitor.cbSize = sizeof(monitor);
  if (!GetMonitorInfoW(MonitorFromWindow(glfwGetWin32Window(window_), MONITOR_DEFAULTTONEAREST), &monitor))
    return false;
  UINT32 path_count{}, mode_count{};
  if (GetDisplayConfigBufferSizes(QDC_ONLY_ACTIVE_PATHS, &path_count, &mode_count) != ERROR_SUCCESS)
    return false;
  std::vector<DISPLAYCONFIG_PATH_INFO> paths(path_count);
  std::vector<DISPLAYCONFIG_MODE_INFO> modes(mode_count);
  if (QueryDisplayConfig(QDC_ONLY_ACTIVE_PATHS, &path_count, paths.data(), &mode_count, modes.data(), nullptr) !=
      ERROR_SUCCESS)
    return false;
  for (UINT32 i = 0; i < path_count; ++i) {
    const auto &path = paths[i];
    DISPLAYCONFIG_SOURCE_DEVICE_NAME source{};
    source.header = {DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME, sizeof(source), path.sourceInfo.adapterId,
                     path.sourceInfo.id};
    if (DisplayConfigGetDeviceInfo(&source.header) != ERROR_SUCCESS ||
        wcscmp(source.viewGdiDeviceName, monitor.szDevice))
      continue;
    DISPLAYCONFIG_GET_ADVANCED_COLOR_INFO color{};
    color.header = {DISPLAYCONFIG_DEVICE_INFO_GET_ADVANCED_COLOR_INFO, sizeof(color), path.targetInfo.adapterId,
                    path.targetInfo.id};
    if (DisplayConfigGetDeviceInfo(&color.header) != ERROR_SUCCESS || !color.advancedColorEnabled)
      return false;
    DISPLAYCONFIG_SDR_WHITE_LEVEL white{};
    white.header = {DISPLAYCONFIG_DEVICE_INFO_GET_SDR_WHITE_LEVEL, sizeof(white), path.targetInfo.adapterId,
                    path.targetInfo.id};
    if (DisplayConfigGetDeviceInfo(&white.header) != ERROR_SUCCESS || !white.SDRWhiteLevel)
      return false;
    nits = 80.0f * (white.SDRWhiteLevel / 1000.0f);
    return true;
  }
#endif
  return false;
}

ColorSpace VulkanWindowImpl::RequestedColorSpace() const {
  return requested_color_space_;
}

void VulkanWindowImpl::DestroyWindow() {
  imgui_.reset();
  glfw::UnregisterWindow(window_);
  glfw::DestroyWindow(window_);
}

bool VulkanWindowImpl::InitializeWindow(IVulkanDevice *device, const WindowDesc &desc) {
  queue_ = device->GetNativeQueue(QUEUE_TYPE_MAIN);
  requested_mode_ = desc.present_mode;
  requested_color_space_ = desc.color_space;
  window_ = glfw::CreateGLFWWindow(desc.width, desc.height, desc.title.c_str());
  if (!window_)
    return false;
  glfw::RegisterWindow(this, this);
  return true;
}

GLFWwindow *VulkanWindowImpl::GLFWWindow() const {
  return window_;
}

bool VulkanWindowImpl::ShouldClose() const {
#if !defined(NDEBUG)
  if (!glfw::IsWindowThread())
    return true;
#endif
  return glfwWindowShouldClose(window_);
}

bool VulkanWindowImpl::IsKeyDown(int key) const {
#if !defined(NDEBUG)
  if (!glfw::IsWindowThread())
    return false;
#endif
  return glfwGetKey(window_, key) == GLFW_PRESS;
}

bool VulkanWindowImpl::SetTitle(const std::string &title) {
  return glfw::SetWindowTitle(window_, title.c_str());
}

EventManager<void(int, int)> &VulkanWindowImpl::ResizeEvent() {
  return resize_event_;
}

EventManager<void(int, int)> &VulkanWindowImpl::FramebufferResizeEvent() {
  return framebufferresize_event_;
}

EventManager<void(bool)> &VulkanWindowImpl::CursorEnterEvent() {
  return cursorenter_event_;
}

EventManager<void(bool)> &VulkanWindowImpl::FocusEvent() {
  return focus_event_;
}

EventManager<void(double, double)> &VulkanWindowImpl::MouseMoveEvent() {
  return mousemove_event_;
}

EventManager<void(int, int, int, double, double)> &VulkanWindowImpl::MouseButtonEvent() {
  return mousebutton_event_;
}

EventManager<void(double, double)> &VulkanWindowImpl::ScrollEvent() {
  return scroll_event_;
}

EventManager<void(int, int, int, int)> &VulkanWindowImpl::KeyEvent() {
  return key_event_;
}

EventManager<void(uint32_t)> &VulkanWindowImpl::CharEvent() {
  return char_event_;
}

EventManager<void(int, const char **)> &VulkanWindowImpl::DropEvent() {
  return drop_event_;
}
}  // namespace grassland::rhi::backend::vulkan
