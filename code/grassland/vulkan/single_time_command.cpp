#include "grassland/vulkan/single_time_command.h"

#include "grassland/vulkan/command_buffer.h"

namespace CD::vulkan {
VkResult SingleTimeCommand(const Queue *queue,
                           const CommandPool *command_pool,
                           std::function<void(VkCommandBuffer)> function) {
  VkSubmitInfo submit_info = {};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  return SingleTimeCommand(queue, command_pool, function, submit_info);
}

VkResult SingleTimeCommand(const Queue *queue,
                           const CommandPool *command_pool,
                           std::function<void(VkCommandBuffer)> function,
                           VkSubmitInfo &submit_info) {
  std::unique_ptr<CommandBuffer> command_buffer;
  command_pool->AllocateCommandBuffer(&command_buffer);

  VkCommandBufferBeginInfo begin_info = {};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  vkBeginCommandBuffer(command_buffer->Handle(), &begin_info);

  function(command_buffer->Handle());

  vkEndCommandBuffer(command_buffer->Handle());

  VkCommandBuffer command_buffers[] = {command_buffer->Handle()};

  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = command_buffers;

  VkResult result = vkQueueSubmit(queue->Handle(), 1, &submit_info, nullptr);

  if (result != VK_SUCCESS) {
    SetErrorMessage("Failed to submit command buffer");
    return result;
  }

  result = vkQueueWaitIdle(queue->Handle());

  return result;
}
}  // namespace CD::vulkan
