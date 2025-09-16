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

void AccelerationStructure::PybindClassRegistration(py::classh<AccelerationStructure> &c) {
  c.def("update_instances",
        py::overload_cast<const std::vector<RayTracingInstance> &>(&AccelerationStructure::UpdateInstances),
        py::arg("instances"), "Update instances with RayTracingInstance objects");
  c.def(
      "update_instances",
      [](AccelerationStructure *self, py::list instances_list) {
        std::vector<std::pair<AccelerationStructure *, glm::mat4x3>> instances;
        for (auto item : instances_list) {
          py::tuple pair = item.cast<py::tuple>();
          if (pair.size() != 2) {
            throw std::runtime_error("Each instance must be a pair of (AccelerationStructure, transform)");
          }
          AccelerationStructure *as = pair[0].cast<AccelerationStructure *>();
          py::list transform_list = pair[1].cast<py::list>();

          if (transform_list.size() != 3) {
            throw std::runtime_error("Transform matrix must have 3 rows");
          }
          glm::mat4x3 transform;
          for (int i = 0; i < 3; i++) {
            py::list row = transform_list[i].cast<py::list>();
            if (row.size() != 4) {
              throw std::runtime_error("Transform matrix rows must have 4 columns");
            }
            for (int j = 0; j < 4; j++) {
              transform[i][j] = row[j].cast<float>();
            }
          }
          instances.emplace_back(as, transform);
        }
        return self->UpdateInstances(instances);
      },
      py::arg("instances"), "Update instances with acceleration structure and transform pairs");

  // Wrap make_instance to handle glm::mat4x3 properly
  c.def(
      "make_instance",
      [](AccelerationStructure *self, py::list transform_list, uint32_t instance_id, uint32_t instance_mask,
         uint32_t instance_hit_group_offset, RayTracingInstanceFlag instance_flags) {
        if (transform_list.size() != 3) {
          throw std::runtime_error("Transform matrix must have 3 rows");
        }
        glm::mat4x3 transform;
        for (int i = 0; i < 3; i++) {
          py::list row = transform_list[i].cast<py::list>();
          if (row.size() != 4) {
            throw std::runtime_error("Transform matrix rows must have 4 columns");
          }
          for (int j = 0; j < 4; j++) {
            transform[i][j] = row[j].cast<float>();
          }
        }
        return self->MakeInstance(transform, instance_id, instance_mask, instance_hit_group_offset, instance_flags);
      },
      py::arg("transform"), py::arg("instance_id") = 0, py::arg("instance_mask") = 0xFF,
      py::arg("instance_hit_group_offset") = 0,
      py::arg_v("instance_flags", RAYTRACING_INSTANCE_FLAG_NONE,
                "RayTracingInstanceFlag.RAYTRACING_INSTANCE_FLAG_NONE"),
      "Create a ray tracing instance", py::keep_alive<0, 1>{});
  c.def("__repr__", [](AccelerationStructure *as) { return py::str("AccelerationStructure()"); });
}

}  // namespace grassland::graphics
