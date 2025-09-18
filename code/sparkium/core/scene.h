#pragma once
#include "sparkium/core/core_util.h"

namespace sparkium {

class Scene {
 public:
  Scene(Core *core);

  void Render(Camera *camera, Film *film);

  void AddEntity(Entity *entity);

  void DeleteEntity(Entity *entity);

  void SetEntityActive(Entity *entity, bool active);

  int32_t RegisterLight(Light *light, int custom_index = -1);

  int32_t RegisterInstance(graphics::AccelerationStructure *blas,
                           const glm::mat4x3 &transformation,
                           int32_t hit_group_index,
                           int32_t geometry_data_index,
                           int32_t material_data_index,
                           int32_t custom_index = -1);

  int &LightCustomIndex(int32_t light_index);

  int &InstanceCustomIndex(int32_t instance_index);

  struct HitGroupComparator {
    bool operator()(const InstanceHitGroups &lhs, const InstanceHitGroups &rhs) const {
      return std::tie(lhs.render_group.closest_hit_shader, lhs.render_group.any_hit_shader,
                      lhs.render_group.intersection_shader, lhs.render_group.procedure,
                      lhs.shadow_group.closest_hit_shader, lhs.shadow_group.any_hit_shader,
                      lhs.shadow_group.intersection_shader, lhs.shadow_group.procedure) <
             std::tie(rhs.render_group.closest_hit_shader, rhs.render_group.any_hit_shader,
                      rhs.render_group.intersection_shader, rhs.render_group.procedure,
                      rhs.shadow_group.closest_hit_shader, rhs.shadow_group.any_hit_shader,
                      rhs.shadow_group.intersection_shader, rhs.shadow_group.procedure);
    }
  };

  struct Settings {
    int samples_per_dispatch = 128;
    int max_bounces = 32;
    int alpha_shadow = false;
  } settings;

  struct EntityStatus {
    bool active{true};
    int shader_version{0};
  };

  int32_t RegisterCallableShader(graphics::Shader *callable_shader);
  int32_t RegisterBuffer(graphics::Buffer *buffer);
  int32_t RegisterImage(graphics::Image *image);
  int32_t RegisterHitGroup(const InstanceHitGroups &hit_group);

 private:
  void UpdatePipeline(Camera *camera);
  Core *core_;
  std::unique_ptr<graphics::Shader> raygen_shader_;
  std::unique_ptr<graphics::Shader> default_miss_shader_;
  std::unique_ptr<graphics::Shader> shadow_miss_shader_;
  std::unique_ptr<graphics::RayTracingProgram> rt_program_;
  std::unique_ptr<graphics::AccelerationStructure> tlas_;
  std::unique_ptr<graphics::Buffer> scene_settings_buffer_;
  std::map<Entity *, EntityStatus> entities_;

  std::vector<int32_t> miss_shader_indices_;
  std::vector<int32_t> hit_group_indices_;
  std::vector<int32_t> callable_shader_indices_;

  std::vector<InstanceHitGroups> hit_groups_;
  std::map<InstanceHitGroups, int32_t, HitGroupComparator> hit_group_map_;

  std::vector<graphics::Buffer *> buffers_;
  std::map<graphics::Buffer *, int32_t> buffer_map_;

  std::vector<graphics::Shader *> callable_shaders_;
  std::map<graphics::Shader *, int32_t> callable_shader_map_;

  std::vector<graphics::Image *> sdr_images_;
  std::map<graphics::Image *, int32_t> sdr_image_map_;

  std::vector<graphics::Image *> hdr_images_;
  std::map<graphics::Image *, int32_t> hdr_image_map_;

  bool pipeline_dirty_{true};
  int buffer_capacity_{0};
  int sdr_image_capacity_{0};
  int hdr_image_capacity_{0};

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

  std::unique_ptr<graphics::Sampler> linear_sampler_;
  std::unique_ptr<graphics::Sampler> nearest_sampler_;
};

}  // namespace sparkium
