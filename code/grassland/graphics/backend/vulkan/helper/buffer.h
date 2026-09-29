#pragma once
#include "grassland/graphics/backend/vulkan/helper/native_types.h"

namespace grassland::graphics::backend::vulkan {
#if defined(LONGMARCH_CUDA_RUNTIME)
VkExternalMemoryHandleTypeFlagBits GetDefaultExternalMemoryHandleType();

void CreateExternalBuffer(VkDevice device,
                          std::function<uint32_t(uint32_t, VkMemoryPropertyFlags)> find_memory_type,
                          VkDeviceSize size,
                          VkBufferUsageFlags usage,
                          VkMemoryPropertyFlags properties,
                          VkBuffer &buffer,
                          VkDeviceMemory &bufferMemory);
#endif

}  // namespace grassland::graphics::backend::vulkan
