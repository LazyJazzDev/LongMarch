#include "cao_di/vulkan/command_pool.h"

#include "cao_di/vulkan/command_buffer.h"
#include "cao_di/vulkan/queue.h"

namespace CD::vulkan {

CommandPool::CommandPool(const struct Device *device, VkCommandPool command_pool)
    : device_(device), command_pool_(command_pool) {
}

CommandPool::~CommandPool() {
  vkDestroyCommandPool(device_->Handle(), command_pool_, nullptr);
}

VkResult CommandPool::AllocateCommandBuffer(VkCommandBufferLevel level,
                                            double_ptr<CommandBuffer> pp_command_buffer) const {
  if (!pp_command_buffer) {
    SetErrorMessage("pp_command_buffer is nullptr");
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  VkCommandBufferAllocateInfo allocate_info{};
  allocate_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocate_info.commandPool = command_pool_;
  allocate_info.level = level;
  allocate_info.commandBufferCount = 1;
  VkCommandBuffer command_buffer;

  RETURN_IF_FAILED_VK(vkAllocateCommandBuffers(device_->Handle(), &allocate_info, &command_buffer),
                      "failed to allocate command buffer!");

  pp_command_buffer.construct(this, command_buffer);

  return VK_SUCCESS;
}

VkResult CommandPool::AllocateCommandBuffer(double_ptr<CommandBuffer> pp_command_buffer) const {
  return AllocateCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, pp_command_buffer);
}

VkResult CommandPool::SingleTimeCommands(Queue *queue, const std::function<void(VkCommandBuffer)> &tasks) const {
  std::unique_ptr<CommandBuffer> command_buffer;
  RETURN_IF_FAILED_VK(AllocateCommandBuffer(&command_buffer), "failed to allocate command buffer!");

  VkCommandBufferBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  RETURN_IF_FAILED_VK(vkBeginCommandBuffer(command_buffer->Handle(), &begin_info), "failed to begin command buffer!");

  tasks(command_buffer->Handle());

  RETURN_IF_FAILED_VK(vkEndCommandBuffer(command_buffer->Handle()), "failed to end command buffer!");

  VkCommandBuffer command_buffer_handle = command_buffer->Handle();

  VkSubmitInfo submit_info{};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &command_buffer_handle;

  RETURN_IF_FAILED_VK(vkQueueSubmit(queue->Handle(), 1, &submit_info, VK_NULL_HANDLE),
                      "failed to submit command buffer!");

  RETURN_IF_FAILED_VK(queue->WaitIdle(), "failed to wait for queue idle!");

  return VK_SUCCESS;
}

VkResult CommandPool::SingleTimeCommands(const std::function<void(VkCommandBuffer)> &tasks) const {
  std::unique_ptr<Queue> queue;
  RETURN_IF_FAILED_VK(device_->GetQueue(device_->PhysicalDevice().GraphicsFamilyIndex(), 0, &queue),
                      "failed to get graphics queue!");
  return SingleTimeCommands(queue.get(), tasks);
}
}  // namespace CD::vulkan
