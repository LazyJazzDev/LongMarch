#pragma once
#include "grassland/vulkan/instance.h"

namespace grassland::vulkan {
class Queue {
 public:
  explicit Queue(const class Device *device,
                 uint32_t queue_family_index,
                 VkQueue queue);

  VkQueue Handle() const;

  const class Device *Device() const {
    return device_;
  }
  uint32_t QueueFamilyIndex() const {
    return queue_family_index_;
  }

  VkResult WaitIdle() const;

 private:
  const class Device *device_{};
  uint32_t queue_family_index_{};
  VkQueue queue_{};
};
}  // namespace grassland::vulkan
