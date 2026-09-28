#include "grassland/graphics/backend/vulkan/helper/raytracing/acceleration_structure.h"

#include "grassland/graphics/backend/vulkan/helper/single_time_command.h"
#include "grassland/graphics/backend/vulkan/vulkan_core.h"

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
                                    VkDeviceSize *buffer_size) {
  VkAccelerationStructureBuildGeometryInfoKHR build_info{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
  build_info.type = type;
  build_info.flags = flags;
  build_info.mode = mode;
  build_info.geometryCount = 1;
  build_info.pGeometries = &geometry;
  if (mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR && *as) {
    build_info.srcAccelerationStructure = *as;
    build_info.dstAccelerationStructure = *as;
  }
  VkAccelerationStructureBuildSizesInfoKHR sizes{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
  core->Procedures().vkGetAccelerationStructureBuildSizesKHR(
      core->Handle(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &build_info, &primitive_count, &sizes);
  if (!*buffer || *buffer_size < sizes.accelerationStructureSize) {
    if (*as)
      core->Procedures().vkDestroyAccelerationStructureKHR(core->Handle(), *as, nullptr);
    if (*buffer)
      vmaDestroyBuffer(core->Allocator(), *buffer, *allocation);
    *as = VK_NULL_HANDLE;
    *buffer = VK_NULL_HANDLE;
    VkResult result = core->CreateBuffer(
        sizes.accelerationStructureSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY, buffer, allocation);
    if (result != VK_SUCCESS)
      return result;
    *buffer_size = sizes.accelerationStructureSize;
    VkAccelerationStructureCreateInfoKHR create_info{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
    create_info.buffer = *buffer;
    create_info.size = sizes.accelerationStructureSize;
    create_info.type = type;
    result = core->Procedures().vkCreateAccelerationStructureKHR(core->Handle(), &create_info, nullptr, as);
    if (result != VK_SUCCESS)
      return result;
    build_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    build_info.srcAccelerationStructure = VK_NULL_HANDLE;
  }
  build_info.dstAccelerationStructure = *as;
  VkPhysicalDeviceAccelerationStructurePropertiesKHR properties{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
  VkPhysicalDeviceProperties2 device_properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  device_properties.pNext = &properties;
  vkGetPhysicalDeviceProperties2(core->PhysicalDevice(), &device_properties);
  VkDeviceSize alignment = properties.minAccelerationStructureScratchOffsetAlignment;
  VkBuffer scratch = VK_NULL_HANDLE;
  VmaAllocation scratch_allocation = VK_NULL_HANDLE;
  VkResult result = core->CreateBuffer(sizes.buildScratchSize + alignment - 1,
                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                       VMA_MEMORY_USAGE_GPU_ONLY, &scratch, &scratch_allocation);
  if (result != VK_SUCCESS)
    return result;
  build_info.scratchData.deviceAddress = (core->BufferAddress(scratch) + alignment - 1) & ~(alignment - 1);
  VkAccelerationStructureBuildRangeInfoKHR range{};
  range.primitiveCount = primitive_count;
  const VkAccelerationStructureBuildRangeInfoKHR *ranges[] = {&range};
  result = SingleTimeCommand(core->Handle(), queue, command_pool, [&](VkCommandBuffer command_buffer) {
    core->Procedures().vkCmdBuildAccelerationStructuresKHR(command_buffer, 1, &build_info, ranges);
  });
  vmaDestroyBuffer(core->Allocator(), scratch, scratch_allocation);
  return result;
}
}  // namespace grassland::graphics::backend::vulkan
