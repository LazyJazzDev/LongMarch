#include "cao_di/graphics/backend/vulkan/vulkan_acceleration_structure.h"

#include "cao_di/graphics/backend/vulkan/vulkan_core.h"

namespace CD::graphics::backend {

VulkanAccelerationStructure::VulkanAccelerationStructure(
    VulkanCore *core,
    std::unique_ptr<vulkan::AccelerationStructure> &&acceleration_structure)
    : core_(core), acceleration_structure_(std::move(acceleration_structure)) {
}

int VulkanAccelerationStructure::UpdateInstances(const std::vector<RayTracingInstance> &instances) {
  std::vector<VkAccelerationStructureInstanceKHR> vulkan_instances;
  vulkan_instances.reserve(instances.size());
  for (const auto &instance : instances) {
    vulkan_instances.emplace_back(RayTracingInstanceToVkAccelerationStructureInstanceKHR(instance));
  }
  acceleration_structure_->UpdateInstances(vulkan_instances, core_->GraphicsCommandPool(), core_->GraphicsQueue());
  return 0;
}
}  // namespace CD::graphics::backend
