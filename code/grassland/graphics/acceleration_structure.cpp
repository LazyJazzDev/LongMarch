#include "grassland/graphics/acceleration_structure.h"

namespace grassland::graphics {

int AccelerationStructure::UpdateInstances(
    const std::vector<std::pair<AccelerationStructure *, glm::mat4x3>> &instances) {
  std::vector<RayTracingInstance> raytracing_instances;
  raytracing_instances.reserve(instances.size());
  for (int i = 0; i < instances.size(); i++) {
    auto &instance = instances[i];
    raytracing_instances.emplace_back(instance.first->MakeInstance(instance.second, i));
  }
  return UpdateInstances(raytracing_instances);
}

RayTracingInstance AccelerationStructure::MakeInstance(const glm::mat4x3 &transform,
                                                       uint32_t instance_id,
                                                       uint32_t instance_mask,
                                                       uint32_t instance_hit_group_offset,
                                                       RayTracingInstanceFlag instance_flags) {
  RayTracingInstance instance;
  instance.acceleration_structure = this;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 4; j++) {
      instance.transform[i][j] = transform[j][i];
    }
  }
  instance.instance_id = instance_id;
  instance.instance_mask = instance_mask;
  instance.instance_hit_group_offset = instance_hit_group_offset;
  instance.instance_flags = instance_flags;
  return instance;
}

void AccelerationStructure::PybindModuleRegistration(py::module_ &m) {
}

}  // namespace grassland::graphics
