#include "grassland/graphics/backend/vulkan/vulkan_acceleration_structure.h"

#include "grassland/graphics/backend/vulkan/helper/raytracing/acceleration_structure.h"
#include "grassland/graphics/backend/vulkan/vulkan_core.h"

namespace grassland::graphics::backend {
VulkanAccelerationStructure::VulkanAccelerationStructure(VulkanCore *core,
                                                         VkAccelerationStructureKHR as,
                                                         VkBuffer buffer,
                                                         VmaAllocation allocation,
                                                         VkDeviceSize buffer_size,
                                                         VkDeviceAddress device_address,
                                                         int num_instance)
    : core_(core),
      as_(as),
      buffer_(buffer),
      allocation_(allocation),
      buffer_size_(buffer_size),
      device_address_(device_address),
      num_instance_(num_instance) {
}

VulkanAccelerationStructure::~VulkanAccelerationStructure() {
  if (as_)
    core_->Procedures().vkDestroyAccelerationStructureKHR(core_->Handle(), as_, nullptr);
  if (buffer_)
    vmaDestroyBuffer(core_->Allocator(), buffer_, allocation_);
}

int VulkanAccelerationStructure::UpdateInstances(const std::vector<RayTracingInstance> &instances) {
  std::vector<VkAccelerationStructureInstanceKHR> vk_instances;
  vk_instances.reserve(instances.size());
  for (const auto &instance : instances)
    vk_instances.emplace_back(RayTracingInstanceToVkAccelerationStructureInstanceKHR(instance));
  VkBuffer instances_buffer = VK_NULL_HANDLE;
  VmaAllocation instances_allocation = VK_NULL_HANDLE;
  VkDeviceSize data_size = sizeof(VkAccelerationStructureInstanceKHR) * std::max(size_t(1), instances.size());
  VkResult result = core_->CreateBuffer(
      data_size,
      VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
      VMA_MEMORY_USAGE_CPU_TO_GPU, 0, 16, &instances_buffer, &instances_allocation);
  if (result != VK_SUCCESS)
    return -1;
  void *mapped = nullptr;
  result = vmaMapMemory(core_->Allocator(), instances_allocation, &mapped);
  if (result != VK_SUCCESS) {
    vmaDestroyBuffer(core_->Allocator(), instances_buffer, instances_allocation);
    return -1;
  }
  if (!instances.empty())
    std::memcpy(mapped, vk_instances.data(), vk_instances.size() * sizeof(vk_instances[0]));
  vmaUnmapMemory(core_->Allocator(), instances_allocation);
  VkAccelerationStructureGeometryKHR geometry{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
  geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
  geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
  geometry.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
  geometry.geometry.instances.arrayOfPointers = VK_FALSE;
  geometry.geometry.instances.data.deviceAddress = core_->BufferAddress(instances_buffer);
  auto mode = num_instance_ == instances.size() ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR
                                                : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
  result = vulkan::BuildAccelerationStructure(
      core_, geometry, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
      VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR,
      mode, static_cast<uint32_t>(instances.size()), core_->GraphicsCommandPool(), core_->GraphicsQueue(), &as_,
      &buffer_, &allocation_, &buffer_size_);
  vmaDestroyBuffer(core_->Allocator(), instances_buffer, instances_allocation);
  if (result != VK_SUCCESS)
    return -1;
  num_instance_ = static_cast<int>(instances.size());
  VkAccelerationStructureDeviceAddressInfoKHR address_info{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
  address_info.accelerationStructure = as_;
  device_address_ = core_->Procedures().vkGetAccelerationStructureDeviceAddressKHR(core_->Handle(), &address_info);
  return 0;
}
}  // namespace grassland::graphics::backend
