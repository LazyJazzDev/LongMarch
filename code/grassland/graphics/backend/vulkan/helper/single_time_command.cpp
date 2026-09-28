#include "grassland/graphics/backend/vulkan/helper/single_time_command.h"

namespace grassland::graphics::backend::vulkan {
VkResult SingleTimeCommand(VkDevice device,
                           VkQueue queue,
                           VkCommandPool pool,
                           const std::function<void(VkCommandBuffer)> &function) {
  VkSubmitInfo submit_info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  return SingleTimeCommand(device, queue, pool, function, submit_info);
}

VkResult SingleTimeCommand(VkDevice device,
                           VkQueue queue,
                           VkCommandPool pool,
                           const std::function<void(VkCommandBuffer)> &function,
                           VkSubmitInfo &submit_info) {
  VkCommandBufferAllocateInfo allocate_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  allocate_info.commandPool = pool;
  allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocate_info.commandBufferCount = 1;
  VkCommandBuffer buffer = VK_NULL_HANDLE;
  RETURN_IF_FAILED_VK(vkAllocateCommandBuffers(device, &allocate_info, &buffer), "Failed to allocate command buffer");
  VkCommandBufferBeginInfo begin_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  VkResult result = vkBeginCommandBuffer(buffer, &begin_info);
  if (result == VK_SUCCESS) {
    function(buffer);
    result = vkEndCommandBuffer(buffer);
  }
  if (result == VK_SUCCESS) {
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &buffer;
    result = vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE);
  }
  if (result == VK_SUCCESS) {
    result = vkQueueWaitIdle(queue);
  }
  vkFreeCommandBuffers(device, pool, 1, &buffer);
  return result;
}
}  // namespace grassland::graphics::backend::vulkan
