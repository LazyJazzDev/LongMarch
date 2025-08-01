#pragma once
#include "sparks/core/core_util.h"

namespace sparks {

class Scene {
 public:
  Scene(Core *core);

  void Render(Camera *camera, Film *film);

  void AddEntity(Entity *entity);

  GeometryRegistration RegisterGeometry(Geometry *geometry);

  SurfaceRegistration RegisterSurface(Surface *surface);

  LightMetadata RegisterLight(Light *light);

  int32_t RegisterInstance(GeometryRegistration geom_reg,
                           const glm::mat4x3 &transformation,
                           SurfaceRegistration surf_reg,
                           LightMetadata light_reg = {});

  struct HitGroupComparator {
    bool operator()(const graphics::HitGroup &lhs, const graphics::HitGroup &rhs) const {
      return std::tie(lhs.closest_hit_shader, lhs.any_hit_shader, lhs.intersection_shader, lhs.procedure) <
             std::tie(rhs.closest_hit_shader, rhs.any_hit_shader, rhs.intersection_shader, rhs.procedure);
    }
  };

  struct Settings {
    int samples_per_dispatch = 128;
    int max_bounces = 32;
  } settings;

  int32_t RegisterCallableShader(graphics::Shader *callable_shader);
  int32_t RegisterBuffer(graphics::Buffer *buffer);

 private:
  void UpdatePipeline(Camera *camera);
  Core *core_;
  std::unique_ptr<graphics::Shader> raygen_shader_;
  std::unique_ptr<graphics::Shader> miss_shader_;
  std::unique_ptr<graphics::RayTracingProgram> rt_program_;
  std::unique_ptr<graphics::AccelerationStructure> tlas_;
  std::unique_ptr<graphics::Buffer> scene_settings_buffer_;
  std::set<Entity *> entities_;

  std::vector<int32_t> miss_shader_indices_;
  std::vector<int32_t> hit_group_indices_;
  std::vector<int32_t> callable_shader_indices_;

  std::vector<graphics::HitGroup> hit_groups_;
  std::map<graphics::HitGroup, int32_t, HitGroupComparator> hit_group_map_;

  std::vector<graphics::Buffer *> buffers_;
  std::map<graphics::Buffer *, int32_t> buffer_map_;

  std::vector<graphics::Shader *> callable_shaders_;
  std::map<graphics::Shader *, int32_t> callable_shader_map_;

  std::vector<graphics::RayTracingInstance> instances_;

  std::vector<InstanceMetadata> instance_metadatas_;
  std::unique_ptr<graphics::Buffer> instance_metadata_buffer_;

  std::unique_ptr<graphics::Buffer> light_selector_buffer_;

  std::vector<LightMetadata> light_metadatas_;
  std::unique_ptr<graphics::Buffer> light_metadatas_buffer_;

  std::vector<BlellochScanMetadata> blelloch_metadatas_;
  std::unique_ptr<graphics::Buffer> blelloch_metadata_buffer_;

  std::unique_ptr<graphics::Shader> gather_light_power_shader_;
  std::unique_ptr<graphics::ComputeProgram> gather_light_power_program_;

  std::unique_ptr<graphics::CommandContext> preprocess_cmd_context_;
};

}  // namespace sparks
