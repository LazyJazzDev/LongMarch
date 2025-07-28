#pragma once
#include "sparks/core/core_util.h"

namespace sparks {

class Scene {
 public:
  Scene(Core *core);

  void Render(Camera *camera, Film *film);

  void AddEntity(Entity *entity);

  GeometryRegistration RegisterGeometry(Geometry *geometry);

  MaterialRegistration RegisterMaterial(Material *material);

  InstanceRegistration RegisterInstance(GeometryRegistration geom_reg,
                                        const glm::mat4 &transformation,
                                        MaterialRegistration mat_reg);

  struct HitGroupComparator {
    bool operator()(const graphics::HitGroup &lhs, const graphics::HitGroup &rhs) const {
      return std::tie(lhs.closest_hit_shader, lhs.any_hit_shader, lhs.intersection_shader, lhs.procedure) <
             std::tie(rhs.closest_hit_shader, rhs.any_hit_shader, rhs.intersection_shader, rhs.procedure);
    }
  };

 private:
  int32_t RegisterCallableShader(graphics::Shader *callable_shader);
  void UpdatePipeline(Camera *camera);
  Core *core_;
  std::unique_ptr<graphics::Shader> raygen_shader_;
  std::unique_ptr<graphics::Shader> miss_shader_;
  std::unique_ptr<graphics::RayTracingProgram> rt_program_;
  std::unique_ptr<graphics::AccelerationStructure> tlas_;
  std::set<Entity *> entities_;

  std::vector<int32_t> miss_shader_indices_;
  std::vector<int32_t> hit_group_indices_;
  std::vector<int32_t> callable_shader_indices_;

  std::vector<graphics::HitGroup> hit_groups_;
  std::map<graphics::HitGroup, int32_t, HitGroupComparator> hit_group_map_;

  std::vector<graphics::Buffer *> geometry_buffers_;
  std::map<graphics::Buffer *, int32_t> geometry_buffer_map_;

  std::vector<graphics::Shader *> callable_shaders_;
  std::map<graphics::Shader *, int32_t> callable_shader_map_;

  std::vector<graphics::Buffer *> material_buffers_;
  std::map<graphics::Buffer *, int32_t> material_buffer_map_;

  std::vector<graphics::RayTracingInstance> instances_;
  std::vector<MaterialRegistration> materials_registrations_;

  std::unique_ptr<graphics::Buffer> mat_reg_buffer_;
};

}  // namespace sparks
