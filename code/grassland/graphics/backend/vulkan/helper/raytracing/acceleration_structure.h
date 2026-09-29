#pragma once
#include "grassland/graphics/backend/vulkan/helper/vulkan_util.h"

namespace grassland::graphics::backend::vulkan {
VkResult BuildAccelerationStructure(const VulkanCore *core,
                                    const VkAccelerationStructureGeometryKHR &geometry,
                                    VkAccelerationStructureTypeKHR type,
                                    VkBuildAccelerationStructureFlagsKHR flags,
                                    VkBuildAccelerationStructureModeKHR mode,
                                    uint32_t primitive_count,
                                    VkCommandPool command_pool,
                                    VkQueue queue,
                                    VkAccelerationStructureKHR *as,
                                    VkBuffer *buffer,
                                    VmaAllocation *allocation,
                                    VkDeviceSize *buffer_size);
}  // namespace grassland::graphics::backend::vulkan
