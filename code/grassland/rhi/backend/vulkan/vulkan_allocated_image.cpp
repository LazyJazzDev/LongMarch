#include "grassland/rhi/backend/vulkan/vulkan_allocated_image.h"

#include <vk_mem_alloc.h>

#include <cstdlib>

#include "grassland/rhi/backend/vulkan/vulkan_device.h"

namespace grassland::rhi::backend::vulkan {
VulkanAllocatedImage::VulkanAllocatedImage(VulkanDevice *device, ImageDesc desc)
    : VulkanImage(desc, device, VK_IMAGE_LAYOUT_GENERAL),
      device_(device) {
}

VulkanAllocatedImage::~VulkanAllocatedImage() {
  if (image_) {
    device_->RenderTargets().Retire(image_);
    views_.reset();
    vmaDestroyImage(device_->Allocator(), image_, allocation_);
  }
}

bool VulkanAllocatedImage::IsValid() const {
  return true;
}

bool VulkanAllocatedImage::SupportsStorage() const {
  return storage_;
}

VkImage VulkanAllocatedImage::NativeImage() const {
  return image_;
}

VulkanAllocatedImage *VulkanAllocatedImage::Create(VulkanDevice *device, const ImageDesc &desc) {
  if (!desc.width || !desc.height)
    return nullptr;
  const auto format = NativeImageFormat(desc.format);
  if (format == VK_FORMAT_UNDEFINED)
    return nullptr;
  try {
    auto image = std::unique_ptr<VulkanAllocatedImage>(new VulkanAllocatedImage(device, desc));
    VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = format;
    info.extent = {desc.width, desc.height, 1};
    info.mipLevels = info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = (IsDepthFormat(desc.format) ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
                                             : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) |
                 VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    if (SupportsImageStorage(device->NativePhysicalDevice(), desc.format))
      info.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
    image->storage_ = (info.usage & VK_IMAGE_USAGE_STORAGE_BIT) != 0;
    VmaAllocationCreateInfo memory{};
    memory.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    if (!Check(vmaCreateImage(device->Allocator(), &info, &memory, &image->image_, &image->allocation_, nullptr),
               "Allocate image"))
      return nullptr;
    image->views_ = Views::Create(device->NativeDevice(), image->image_, desc.format, true);
    if (!image->views_)
      return nullptr;
    // Establish the fixed default layout once, before exposing the resource.
    // Temporary initialization commands are outside the rendering loop.
    const auto native = device->NativeDevice();
    std::lock_guard<std::mutex> lock(device->QueueMutex());
    VkCommandPool pool{};
    const auto status = [&]() -> VkResult {
      VkCommandPoolCreateInfo create{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
      create.queueFamilyIndex = device->QueueFamily();
      auto status = vkCreateCommandPool(native, &create, nullptr, &pool);
      if (status != VK_SUCCESS)
        return status;
      VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
      allocate.commandPool = pool;
      allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
      allocate.commandBufferCount = 1;
      VkCommandBuffer commands{};
      status = vkAllocateCommandBuffers(native, &allocate, &commands);
      if (status != VK_SUCCESS)
        return status;
      VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      status = vkBeginCommandBuffer(commands, &begin);
      if (status != VK_SUCCESS)
        return status;
      VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
      barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
      barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
      barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.image = image->image_;
      barrier.subresourceRange = {ImageAspect(desc.format), 0, 1, 0, 1};
      vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0,
                           nullptr, 0, nullptr, 1, &barrier);
      status = vkEndCommandBuffer(commands);
      if (status != VK_SUCCESS)
        return status;
      const auto semaphore = device->Timeline();
      const auto value = device->NextSerial();
      VkTimelineSemaphoreSubmitInfo timeline{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
      timeline.signalSemaphoreValueCount = 1;
      timeline.pSignalSemaphoreValues = &value;
      VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
      submit.pNext = &timeline;
      submit.signalSemaphoreCount = 1;
      submit.pSignalSemaphores = &semaphore;
      submit.commandBufferCount = 1;
      submit.pCommandBuffers = &commands;
      status = vkQueueSubmit(device->NativeQueue(), 1, &submit, VK_NULL_HANDLE);
      if (status != VK_SUCCESS)
        return status;
      if (!Check(device->Wait(value), "Wait for image initialization"))
        std::abort();
      return VK_SUCCESS;
    }();
    if (pool)
      vkDestroyCommandPool(native, pool, nullptr);
    if (!Check(status, "Initialize image layout"))
      return nullptr;
    return image.release();
  } catch (const std::bad_alloc &) {
    ReportError(-1, "Image allocation ran out of memory.");
    return nullptr;
  }
}
}  // namespace grassland::rhi::backend::vulkan
