#include "grassland/vulkan/queue.h"

#include "grassland/vulkan/device.h"

namespace CD::vulkan {

Queue::Queue(const struct Device *device, uint32_t queue_family_index, VkQueue queue)
    : device_(device), queue_family_index_(queue_family_index), queue_(queue) {
}

VkQueue Queue::Handle() const {
  return queue_;
}

VkResult Queue::WaitIdle() const {
  return vkQueueWaitIdle(queue_);
}
}  // namespace CD::vulkan
