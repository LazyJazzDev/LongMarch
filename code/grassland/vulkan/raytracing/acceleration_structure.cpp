#include "grassland/vulkan/raytracing/acceleration_structure.h"

#include "grassland/vulkan/buffer.h"
#include "grassland/vulkan/command_pool.h"
#include "grassland/vulkan/queue.h"

namespace CD::vulkan {

AccelerationStructure::AccelerationStructure(const class Device *device,
                                             std::unique_ptr<class Buffer> buffer,
                                             VkDeviceAddress device_address,
                                             VkAccelerationStructureKHR as)
    : device_(device), buffer_(std::move(buffer)), device_address_(device_address), as_(as) {
}

AccelerationStructure::~AccelerationStructure() {
  device_->Procedures().vkDestroyAccelerationStructureKHR(device_->Handle(), as_, nullptr);
}

class Buffer *AccelerationStructure::Buffer() const {
  return buffer_.get();
}

VkDeviceAddress AccelerationStructure::DeviceAddress() const {
  return device_address_;
}

VkResult AccelerationStructure::UpdateInstances(const std::vector<VkAccelerationStructureInstanceKHR> &instances,
                                                CommandPool *command_pool,
                                                Queue *queue) {
  std::unique_ptr<class Buffer> instances_buffer;
  device_->CreateBuffer(
      sizeof(VkAccelerationStructureInstanceKHR) * instances.size(),
      VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
      VMA_MEMORY_USAGE_CPU_TO_GPU, &instances_buffer);
  std::memcpy(instances_buffer->Map(), instances.data(), instances.size() * sizeof(VkAccelerationStructureInstanceKHR));
  instances_buffer->Unmap();

  VkDeviceOrHostAddressConstKHR instance_data_device_address{};
  instance_data_device_address.deviceAddress = instances_buffer->GetDeviceAddress();

  // The top level acceleration structure contains (bottom level) instance as
  // the input geometry
  VkAccelerationStructureGeometryKHR acceleration_structure_geometry{};
  acceleration_structure_geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
  acceleration_structure_geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
  acceleration_structure_geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
  acceleration_structure_geometry.geometry.instances.sType =
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
  acceleration_structure_geometry.geometry.instances.arrayOfPointers = VK_FALSE;
  acceleration_structure_geometry.geometry.instances.data = instance_data_device_address;

  BuildAccelerationStructure(
      device_, acceleration_structure_geometry, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
      VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR,
      VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR, instances.size(), command_pool, queue, &as_, &buffer_);

  // Get the top acceleration structure's handle, which will be used to setup
  // it's descriptor
  VkAccelerationStructureDeviceAddressInfoKHR acceleration_device_address_info{};
  acceleration_device_address_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
  acceleration_device_address_info.accelerationStructure = as_;
  device_address_ = device_->Procedures().vkGetAccelerationStructureDeviceAddressKHR(device_->Handle(),
                                                                                     &acceleration_device_address_info);

  return VK_SUCCESS;
}

VkResult AccelerationStructure::UpdateInstances(
    const std::vector<std::pair<AccelerationStructure *, glm::mat4>> &objects,
    CommandPool *command_pool,
    Queue *queue) {
  std::vector<VkAccelerationStructureInstanceKHR> acceleration_structure_instances;
  for (int i = 0; i < objects.size(); i++) {
    auto &object = objects[i];
    VkAccelerationStructureInstanceKHR acceleration_structure_instance{};
    acceleration_structure_instance.transform = {object.second[0][0], object.second[1][0], object.second[2][0],
                                                 object.second[3][0], object.second[0][1], object.second[1][1],
                                                 object.second[2][1], object.second[3][1], object.second[0][2],
                                                 object.second[1][2], object.second[2][2], object.second[3][2]};
    acceleration_structure_instance.instanceCustomIndex = i;
    acceleration_structure_instance.mask = 0xFF;
    acceleration_structure_instance.instanceShaderBindingTableRecordOffset = 0;
    acceleration_structure_instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    acceleration_structure_instance.accelerationStructureReference = object.first->DeviceAddress();
    acceleration_structure_instances.push_back(acceleration_structure_instance);
  }

  return UpdateInstances(acceleration_structure_instances, command_pool, queue);
}

VkResult BuildAccelerationStructure(const Device *device,
                                    VkAccelerationStructureGeometryKHR geometry,
                                    VkAccelerationStructureTypeKHR type,
                                    VkBuildAccelerationStructureFlagsKHR flags,
                                    VkBuildAccelerationStructureModeKHR mode,
                                    uint32_t primitive_count,
                                    CommandPool *command_pool,
                                    Queue *queue,
                                    VkAccelerationStructureKHR *ptr_acceleration_structure,
                                    double_ptr<Buffer> pp_buffer) {
  VkAccelerationStructureBuildGeometryInfoKHR build_geometry_info{};
  build_geometry_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
  build_geometry_info.type = type;
  build_geometry_info.flags = flags;
  build_geometry_info.mode = mode;
  if (mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR && *ptr_acceleration_structure != VK_NULL_HANDLE) {
    build_geometry_info.srcAccelerationStructure = *ptr_acceleration_structure;
    build_geometry_info.dstAccelerationStructure = *ptr_acceleration_structure;
  }
  build_geometry_info.geometryCount = 1;
  build_geometry_info.pGeometries = &geometry;

  VkAccelerationStructureBuildSizesInfoKHR build_sizes_info{};
  build_sizes_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
  device->Procedures().vkGetAccelerationStructureBuildSizesKHR(
      device->Handle(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &build_geometry_info, &primitive_count,
      &build_sizes_info);

  if (!(*pp_buffer) || pp_buffer->Size() < build_sizes_info.accelerationStructureSize) {
    // Create a buffer to hold the acceleration structure
    RETURN_IF_FAILED_VK(device->CreateBuffer(build_sizes_info.accelerationStructureSize,
                                             VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                                                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                             VMA_MEMORY_USAGE_GPU_ONLY, pp_buffer),
                        "failed to create buffer!");

    // Create the acceleration structure
    VkAccelerationStructureCreateInfoKHR create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    create_info.buffer = pp_buffer->Handle();
    create_info.size = build_sizes_info.accelerationStructureSize;
    create_info.type = type;
    RETURN_IF_FAILED_VK(device->Procedures().vkCreateAccelerationStructureKHR(device->Handle(), &create_info, nullptr,
                                                                              ptr_acceleration_structure),
                        "failed to create acceleration structure!");
  }

  // The actual build process starts here

  // Create a scratch buffer as a temporary storage for the acceleration
  // structure build

  VkPhysicalDeviceAccelerationStructurePropertiesKHR acceleration_structure_properties{};
  acceleration_structure_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR;
  VkPhysicalDeviceProperties2 device_properties{};
  device_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  device_properties.pNext = &acceleration_structure_properties;
  vkGetPhysicalDeviceProperties2(device->PhysicalDevice().Handle(), &device_properties);
  VkDeviceSize alignment = acceleration_structure_properties.minAccelerationStructureScratchOffsetAlignment;

  // minAccelerationStructureScratchOffsetAlignment

  std::unique_ptr<Buffer> scratch_buffer;
  RETURN_IF_FAILED_VK(
      device->CreateBuffer(build_sizes_info.buildScratchSize + alignment - 1,
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                           VMA_MEMORY_USAGE_GPU_ONLY, &scratch_buffer),
      "failed to create scratch buffer!");

  build_geometry_info.dstAccelerationStructure = *ptr_acceleration_structure;
  build_geometry_info.scratchData.deviceAddress =
      (scratch_buffer->GetDeviceAddress() + alignment - 1) & (~(alignment - 1));

  VkAccelerationStructureBuildRangeInfoKHR acceleration_structure_build_range_info{};
  acceleration_structure_build_range_info.primitiveCount = primitive_count;
  acceleration_structure_build_range_info.primitiveOffset = 0;
  acceleration_structure_build_range_info.firstVertex = 0;
  acceleration_structure_build_range_info.transformOffset = 0;
  std::vector<VkAccelerationStructureBuildRangeInfoKHR *> acceleration_build_structure_range_infos = {
      &acceleration_structure_build_range_info};

  RETURN_IF_FAILED_VK(command_pool->SingleTimeCommands(queue,
                                                       [&](VkCommandBuffer command_buffer) {
                                                         device->Procedures().vkCmdBuildAccelerationStructuresKHR(
                                                             command_buffer, 1, &build_geometry_info,
                                                             acceleration_build_structure_range_infos.data());
                                                       }),
                      "failed to build acceleration structure!");

  return VK_SUCCESS;
}
}  // namespace CD::vulkan
