#include "Surface.h"

#include <hilog/log.h>

#include <cmath>
#include <limits>
#include <stdexcept>

#include "XEngineCapabilities.h"

using namespace grassland;
namespace vulkan = grassland::graphics::backend::vulkan;

namespace longmarch::harmony {
Surface::Surface(graphics::Core *core, OHNativeWindow *window)
    : core_(dynamic_cast<graphics::backend::VulkanCore *>(core)),
      window_(window) {
  if (!core_ || !window)
    throw std::runtime_error("Missing Vulkan core or native window");
  auto create =
      reinterpret_cast<PFN_vkCreateSurfaceOHOS>(vkGetInstanceProcAddr(core_->Instance(), "vkCreateSurfaceOHOS"));
  if (!create)
    throw std::runtime_error("VK_OHOS_surface is unavailable");
  const auto physical_device = core_->PhysicalDevice();
  const auto properties = vulkan::GetPhysicalDeviceProperties(physical_device);
  OH_LOG_Print(
      LOG_APP, LOG_INFO, 0, "LongMarchGPU",
      "GPU=%{public}s API=%{public}u.%{public}u RTpipeline=%{public}u RayQuery=%{public}u storageBuffers=%{public}u sampledImages=%{public}u",
      properties.deviceName, VK_VERSION_MAJOR(properties.apiVersion), VK_VERSION_MINOR(properties.apiVersion),
      vulkan::SupportRayTracing(physical_device), vulkan::SupportRayQuery(physical_device),
      properties.limits.maxPerStageDescriptorStorageBuffers, properties.limits.maxPerStageDescriptorSampledImages);
  OH_LOG_Print(
      LOG_APP, LOG_INFO, 0, "LongMarchGPU",
      "RT pipeline extension=%{public}u ray query extension=%{public}u acceleration structure extension=%{public}u",
      vulkan::IsExtensionSupported(physical_device, VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME),
      vulkan::IsExtensionSupported(physical_device, VK_KHR_RAY_QUERY_EXTENSION_NAME),
      vulkan::IsExtensionSupported(physical_device, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME));
  LogXEngineCapabilities(physical_device);
  uint32_t layer_count = 0;
  if (vkEnumerateInstanceLayerProperties(&layer_count, nullptr) == VK_SUCCESS) {
    std::vector<VkLayerProperties> layers(layer_count);
    if (vkEnumerateInstanceLayerProperties(&layer_count, layers.data()) == VK_SUCCESS)
      for (const auto &layer : layers)
        OH_LOG_Print(LOG_APP, LOG_INFO, 0, "LongMarchGPU", "Vulkan layer %{public}s", layer.layerName);
  }
  VkSurfaceCreateInfoOHOS info{};
  info.sType = VK_STRUCTURE_TYPE_SURFACE_CREATE_INFO_OHOS;
  info.window = window;
  vulkan::ThrowIfFailed(create(core_->Instance(), &info, nullptr, &surface_), "Create OHOS surface");
  VkBool32 supported = VK_FALSE;
  VkResult result = vkGetPhysicalDeviceSurfaceSupportKHR(
      core_->PhysicalDevice(), vulkan::GraphicsFamilyIndex(core_->PhysicalDevice()), surface_, &supported);
  if (result != VK_SUCCESS || !supported) {
    vkDestroySurfaceKHR(core_->Instance(), surface_, nullptr);
    surface_ = VK_NULL_HANDLE;
    throw std::runtime_error("Graphics queue cannot present to this display");
  }
  auto device = core_->Handle();
  VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pool.queueFamilyIndex = vulkan::GraphicsFamilyIndex(core_->PhysicalDevice());
  vulkan::ThrowIfFailed(vkCreateCommandPool(device, &pool, nullptr, &command_pool_), "Presentation command pool");
  for (auto &frame : frames_) {
    VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocate.commandPool = command_pool_;
    allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocate.commandBufferCount = 1;
    vulkan::ThrowIfFailed(vkAllocateCommandBuffers(device, &allocate, &frame.commands), "Presentation commands");
    VkSemaphoreCreateInfo semaphore{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    vulkan::ThrowIfFailed(vkCreateSemaphore(device, &semaphore, nullptr, &frame.acquired), "Acquire semaphore");
    vulkan::ThrowIfFailed(vkCreateSemaphore(device, &semaphore, nullptr, &frame.blitted), "Blit semaphore");
    VkFenceCreateInfo fence{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fence.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    vulkan::ThrowIfFailed(vkCreateFence(device, &fence, nullptr, &frame.done), "Presentation fence");
  }
}

Surface::~Surface() {
  vkDeviceWaitIdle(core_->Handle());
  ReleaseSwapchain();
  for (auto &frame : frames_) {
    if (frame.done)
      vkDestroyFence(core_->Handle(), frame.done, nullptr);
    if (frame.acquired)
      vkDestroySemaphore(core_->Handle(), frame.acquired, nullptr);
    if (frame.blitted)
      vkDestroySemaphore(core_->Handle(), frame.blitted, nullptr);
  }
  if (command_pool_)
    vkDestroyCommandPool(core_->Handle(), command_pool_, nullptr);
  if (surface_)
    vkDestroySurfaceKHR(core_->Instance(), surface_, nullptr);
}

void Surface::ReleaseSwapchain() {
  if (swapchain_)
    vkDestroySwapchainKHR(core_->Handle(), swapchain_, nullptr);
  swapchain_ = VK_NULL_HANDLE;
  images_.clear();
}

void Surface::Resize(uint32_t width, uint32_t height, bool hdr) {
  if (!width || !height)
    return;
  if (swapchain_ && width == requested_width_ && height == requested_height_ && hdr == requested_hdr_)
    return;
  core_->WaitGPU();
  ReleaseSwapchain();
  requested_width_ = width;
  requested_height_ = height;
  requested_hdr_ = hdr;
  auto device = core_->Handle();
  auto physical = core_->PhysicalDevice();
  VkSurfaceCapabilitiesKHR caps{};
  vulkan::ThrowIfFailed(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical, surface_, &caps), "Surface capabilities");
  if (!(caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT))
    throw std::runtime_error("Display does not support Vulkan transfer presentation");
  uint32_t count = 0;
  vulkan::ThrowIfFailed(vkGetPhysicalDeviceSurfaceFormatsKHR(physical, surface_, &count, nullptr), "Surface formats");
  std::vector<VkSurfaceFormatKHR> formats(count);
  vulkan::ThrowIfFailed(vkGetPhysicalDeviceSurfaceFormatsKHR(physical, surface_, &count, formats.data()),
                        "Surface formats");
  if (formats.empty())
    throw std::runtime_error("Display has no Vulkan surface formats");
  auto format = formats.front();
  for (auto candidate : formats) {
    if ((candidate.format == VK_FORMAT_B8G8R8A8_UNORM || candidate.format == VK_FORMAT_R8G8B8A8_UNORM) &&
        candidate.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
      format = candidate;
  }
  hdr_ = false;
  if (hdr) {
    for (auto candidate : formats) {
      // Some OHOS drivers advertise 10-bit UNORM only with sRGB/P3 WSI color
      // spaces. The public native-window API carries the actual BT.2020/PQ
      // buffer metadata to RenderService independently of this WSI selection.
      if (candidate.format == VK_FORMAT_A2B10G10R10_UNORM_PACK32 &&
          (candidate.colorSpace == VK_COLOR_SPACE_HDR10_ST2084_EXT ||
           candidate.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)) {
        format = candidate;
        hdr_ = true;
        break;
      }
    }
  }
  VkFormatProperties properties{};
  vkGetPhysicalDeviceFormatProperties(physical, format.format, &properties);
  if (!(properties.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT))
    throw std::runtime_error("Display format does not support image blits");
  extent_ = caps.currentExtent;
  if (extent_.width == UINT32_MAX) {
    extent_.width = std::clamp(width, caps.minImageExtent.width, caps.maxImageExtent.width);
    extent_.height = std::clamp(height, caps.minImageExtent.height, caps.maxImageExtent.height);
  }
  VkSwapchainCreateInfoKHR info{};
  info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  info.surface = surface_;
  info.minImageCount = std::max(2u, caps.minImageCount);
  if (caps.maxImageCount)
    info.minImageCount = std::min(info.minImageCount, caps.maxImageCount);
  info.imageFormat = format.format;
  info.imageColorSpace = format.colorSpace;
  info.imageExtent = extent_;
  info.imageArrayLayers = 1;
  info.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  // Render in ArkUI's current orientation; let the compositor transform it.
  info.preTransform = (caps.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR)
                          ? VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR
                          : caps.currentTransform;
  info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  for (auto alpha : {VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR, VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
                     VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR, VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR}) {
    if (caps.supportedCompositeAlpha & alpha) {
      info.compositeAlpha = alpha;
      break;
    }
  }
  info.presentMode = VK_PRESENT_MODE_FIFO_KHR;
  info.clipped = VK_TRUE;
  vulkan::ThrowIfFailed(vkCreateSwapchainKHR(device, &info, nullptr, &swapchain_), "Create swapchain");
  const auto color_space = hdr_ ? OH_COLORSPACE_BT2020_PQ_FULL : OH_COLORSPACE_SRGB_FULL;
  auto metadata_type = hdr_ ? OH_VIDEO_HDR_HDR10 : OH_VIDEO_NONE;
  OH_NativeBuffer_StaticMetadata metadata{};
  if (hdr_) {
    metadata.smpte2086.displayPrimaryRed = {0.708f, 0.292f};
    metadata.smpte2086.displayPrimaryGreen = {0.170f, 0.797f};
    metadata.smpte2086.displayPrimaryBlue = {0.131f, 0.046f};
    metadata.smpte2086.whitePoint = {0.3127f, 0.3290f};
    metadata.smpte2086.maxLuminance = 1000.0f;
    metadata.smpte2086.minLuminance = 0.0001f;
    // Bounds of the encoded signal, not measurements of the attached panel.
    metadata.cta861.maxContentLightLevel = 1000.0f;
    metadata.cta861.maxFrameAverageLightLevel = 1000.0f;
  }
  const int color_result = OH_NativeWindow_SetColorSpace(window_, color_space);
  const int type_result = OH_NativeWindow_SetMetadataValue(window_, OH_HDR_METADATA_TYPE, sizeof(metadata_type),
                                                           reinterpret_cast<uint8_t *>(&metadata_type));
  const int static_result = OH_NativeWindow_SetMetadataValue(window_, OH_HDR_STATIC_METADATA, sizeof(metadata),
                                                             reinterpret_cast<uint8_t *>(&metadata));
  OH_LOG_Print(LOG_APP, LOG_INFO, 0, "LongMarchHDR",
               "HDR10=%{public}d format=%{public}d nativeColor=%{public}d results=%{public}d/%{public}d/%{public}d",
               int(hdr_), int(format.format), int(color_space), color_result, type_result, static_result);
  if (hdr_ && (color_result || type_result || static_result))
    throw std::runtime_error("Cannot configure native HDR10 color space/metadata");
  vulkan::ThrowIfFailed(vkGetSwapchainImagesKHR(device, swapchain_, &count, nullptr), "Swapchain images");
  images_.resize(count);
  vulkan::ThrowIfFailed(vkGetSwapchainImagesKHR(device, swapchain_, &count, images_.data()), "Swapchain images");
}

bool Surface::Present(graphics::Image *source,
                      double zoom,
                      double pan_x,
                      double pan_y,
                      bool linear_demo,
                      bool encoded_particles) {
  if (!swapchain_ || !source)
    return false;
  if (hdr_ || linear_demo || encoded_particles) {
    if (!presentation_)
      presentation_ = std::make_unique<HdrPresentation>(core_);
    source = presentation_->Convert(source, hdr_, encoded_particles);
  }
  auto *image = dynamic_cast<graphics::backend::VulkanImage *>(source);
  if (!image)
    throw std::runtime_error("Presentation requires a Vulkan image");
  auto device = core_->Handle();
  auto &frame = frames_[frame_];
  // Reuse this slot only after its previous blit, which bounds work in flight.
  vulkan::ThrowIfFailed(vkWaitForFences(device, 1, &frame.done, VK_TRUE, UINT64_MAX), "Wait presentation");
  uint32_t index = 0;
  VkResult acquire = vkAcquireNextImageKHR(device, swapchain_, UINT64_MAX, frame.acquired, VK_NULL_HANDLE, &index);
  if (acquire == VK_ERROR_OUT_OF_DATE_KHR) {
    core_->WaitGPU();
    ReleaseSwapchain();
    Resize(requested_width_, requested_height_, requested_hdr_);
    return false;
  }
  if (acquire != VK_SUBOPTIMAL_KHR)
    vulkan::ThrowIfFailed(acquire, "Acquire display image");
  vulkan::ThrowIfFailed(vkResetFences(device, 1, &frame.done), "Reset presentation fence");
  frame_ = (frame_ + 1) % kFramesInFlight;
  auto size = image->Extent();
  const double scale = std::min(double(extent_.width) / size.width, double(extent_.height) / size.height) * zoom;
  const double left = (extent_.width - size.width * scale) / 2 + pan_x;
  const double top = (extent_.height - size.height * scale) / 2 + pan_y;
  VkImageBlit blit{};
  blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  blit.dstSubresource = blit.srcSubresource;
  auto clip = [](double value, uint32_t bound) { return int32_t(std::clamp(value, 0.0, double(bound))); };
  blit.dstOffsets[0] = {clip(left, extent_.width), clip(top, extent_.height), 0};
  blit.dstOffsets[1] = {clip(left + size.width * scale, extent_.width), clip(top + size.height * scale, extent_.height),
                        1};
  blit.srcOffsets[0] = {clip((blit.dstOffsets[0].x - left) / scale, size.width),
                        clip((blit.dstOffsets[0].y - top) / scale, size.height), 0};
  blit.srcOffsets[1] = {clip((blit.dstOffsets[1].x - left) / scale, size.width),
                        clip((blit.dstOffsets[1].y - top) / scale, size.height), 1};
  {
    VkCommandBuffer cmd = frame.commands;
    vkResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vulkan::ThrowIfFailed(vkBeginCommandBuffer(cmd, &begin), "Begin presentation commands");
    auto transit = [&](VkImage target, VkImageLayout before, VkImageLayout after, VkAccessFlags src,
                       VkAccessFlags dst) {
      vulkan::TransitImageLayout(cmd, target, before, after, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                 VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, src, dst, VK_IMAGE_ASPECT_COLOR_BIT);
    };
    transit(images_[index], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
            VK_ACCESS_TRANSFER_WRITE_BIT);
    VkClearColorValue black{};
    black.float32[3] = 1;
    VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage(cmd, images_[index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &range);
    transit(images_[index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    transit(image->Handle(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_MEMORY_WRITE_BIT,
            VK_ACCESS_TRANSFER_READ_BIT);
    if (blit.dstOffsets[1].x > blit.dstOffsets[0].x && blit.dstOffsets[1].y > blit.dstOffsets[0].y &&
        blit.srcOffsets[1].x > blit.srcOffsets[0].x && blit.srcOffsets[1].y > blit.srcOffsets[0].y) {
      vkCmdBlitImage(cmd, image->Handle(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, images_[index],
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_NEAREST);
    }
    transit(image->Handle(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_READ_BIT,
            VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
    transit(images_[index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            VK_ACCESS_TRANSFER_WRITE_BIT, 0);
    vulkan::ThrowIfFailed(vkEndCommandBuffer(cmd), "End presentation commands");
    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.waitSemaphoreCount = 1;
    submit.pWaitSemaphores = &frame.acquired;
    submit.pWaitDstStageMask = &wait_stage;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &frame.blitted;
    vulkan::ThrowIfFailed(vkQueueSubmit(core_->GraphicsQueue(), 1, &submit, frame.done), "Submit presentation");
  }
  VkPresentInfoKHR info{};
  info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  info.waitSemaphoreCount = 1;
  info.pWaitSemaphores = &frame.blitted;
  info.swapchainCount = 1;
  info.pSwapchains = &swapchain_;
  info.pImageIndices = &index;
  VkResult result = vkQueuePresentKHR(core_->GraphicsQueue(), &info);
  if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
    core_->WaitGPU();
    ReleaseSwapchain();
    return false;
  }
  vulkan::ThrowIfFailed(result, "Present display image");
  return true;
}
}  // namespace longmarch::harmony
