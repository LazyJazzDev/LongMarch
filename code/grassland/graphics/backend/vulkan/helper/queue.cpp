#include "grassland/graphics/backend/vulkan/helper/queue.h"

#include "grassland/graphics/backend/vulkan/helper/device.h"

namespace grassland::graphics::backend::vulkan {

Queue::Queue(const struct Device *device, uint32_t queue_family_index, VkQueue queue)
    : device_(device),
      queue_family_index_(queue_family_index),
      queue_(queue) {
}

VkQueue Queue::Handle() const {
  return queue_;
}

VkResult Queue::WaitIdle() const {
  return vkQueueWaitIdle(queue_);
}
}  // namespace grassland::graphics::backend::vulkan
