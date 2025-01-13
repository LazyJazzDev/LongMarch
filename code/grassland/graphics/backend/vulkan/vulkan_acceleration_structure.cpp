#include "grassland/graphics/backend/vulkan/vulkan_acceleration_structure.h"

#include "grassland/graphics/backend/vulkan/vulkan_core.h"

namespace grassland::graphics::backend {

VulkanAccelerationStructure::VulkanAccelerationStructure(
    VulkanCore *core,
    std::unique_ptr<vulkan::AccelerationStructure> &&acceleration_structure)
    : core_(core), acceleration_structure_(std::move(acceleration_structure)) {
}

int VulkanAccelerationStructure::UpdateInstances(
    const std::vector<std::pair<AccelerationStructure *, glm::mat4>> &instances) {
  std::vector<std::pair<vulkan::AccelerationStructure *, glm::mat4>> vulkan_instances;
  vulkan_instances.reserve(instances.size());
  for (const auto &instance : instances) {
    auto vk_as = dynamic_cast<VulkanAccelerationStructure *>(instance.first);
    assert(vk_as != nullptr);
    vulkan_instances.emplace_back(vk_as->acceleration_structure_.get(), instance.second);
  }
  acceleration_structure_->UpdateInstances(vulkan_instances, core_->GraphicsCommandPool(), core_->GraphicsQueue());
  return 0;
}
}  // namespace grassland::graphics::backend
