#pragma once
#include "sparkium/pipelines/raytracing/core/core_util.h"
#include "sparkium/pipelines/raytracing/core/software_pipeline.h"

namespace sparkium::raytracing {

class Scene : public Object {
 public:
  Scene(sparkium::Scene &scene);

  void Render(Camera *camera, Film *film, bool software = false, bool ray_query = false);

  bool SoftwareTracing() const {
    return software_tracing_;
  }

  int32_t RegisterSoftwareInstance(Geometry *geometry,
                                   Material *material,
                                   const glm::mat4x3 &transform,
                                   int32_t custom_index);

  int32_t RegisterLight(Light *light, int custom_index = -1);

  int32_t RegisterInstance(rhi::IBottomLevelAccelerationStructure *blas,
                           const glm::mat4x3 &transformation,
                           int32_t hit_group_index,
                           int32_t geometry_data_index,
                           int32_t material_data_index,
                           int32_t custom_index = -1);

  int &LightCustomIndex(int32_t light_index);

  int &InstanceCustomIndex(int32_t instance_index);

  struct HitGroupComparator {
    bool operator()(const InstanceHitGroups &lhs, const InstanceHitGroups &rhs) const {
      return std::tie(lhs.render_group.closest_hit, lhs.render_group.any_hit, lhs.render_group.intersection,
                      lhs.shadow_group.closest_hit, lhs.shadow_group.any_hit, lhs.shadow_group.intersection) <
             std::tie(rhs.render_group.closest_hit, rhs.render_group.any_hit, rhs.render_group.intersection,
                      rhs.shadow_group.closest_hit, rhs.shadow_group.any_hit, rhs.shadow_group.intersection);
    }
  };

  using Settings = sparkium::Scene::Settings;
  Settings &settings;

  struct EntityStatus {
    bool active{true};
    bool keep{false};
    int shader_version{0};
  };

  int32_t RegisterCallableShader(rhi::IShader *callable_shader);
  int32_t RegisterBuffer(rhi::IBuffer *buffer);
  int32_t RegisterImage(rhi::IImage *image);
  int32_t RegisterHitGroup(const InstanceHitGroups &hit_group);

 private:
  void UpdatePipeline(Camera *camera);
  bool software_tracing_{false};
  bool ray_query_{false};
  bool rendered_{false};
  std::unique_ptr<SoftwarePipeline> software_pipeline_;
  sparkium::Scene &scene_;
  Core *core_;

  rhi::ObjPtr<rhi::IShader> raygen_shader_;
  rhi::ObjPtr<rhi::IShader> default_miss_shader_;
  rhi::ObjPtr<rhi::IShader> shadow_miss_shader_;
  rhi::ObjPtr<rhi::IRayTracingProgram> rt_program_;
  rhi::ObjPtr<rhi::ITopLevelAccelerationStructure> tlas_;
  rhi::ObjPtr<rhi::IBuffer> scene_settings_buffer_;
  std::map<Entity *, EntityStatus> entities_;

  std::vector<InstanceHitGroups> hit_groups_;
  std::map<InstanceHitGroups, int32_t, HitGroupComparator> hit_group_map_;

  std::vector<rhi::IBuffer *> buffers_;
  std::map<rhi::IBuffer *, int32_t> buffer_map_;

  std::vector<rhi::IShader *> callable_shaders_;
  std::map<rhi::IShader *, int32_t> callable_shader_map_;

  std::vector<rhi::IImage *> sdr_images_;
  std::map<rhi::IImage *, int32_t> sdr_image_map_;

  std::vector<rhi::IImage *> hdr_images_;
  std::map<rhi::IImage *, int32_t> hdr_image_map_;

  bool pipeline_dirty_{true};
  int buffer_capacity_{0};
  int sdr_image_capacity_{0};
  int hdr_image_capacity_{0};

  std::vector<rhi::RayTracingInstance> instances_;

  std::vector<InstanceMetadata> instance_metadatas_;
  rhi::ObjPtr<rhi::IBuffer> instance_metadata_buffer_;

  rhi::ObjPtr<rhi::IBuffer> light_selector_buffer_;

  std::vector<LightMetadata> light_metadatas_;
  rhi::ObjPtr<rhi::IBuffer> light_metadatas_buffer_;

  std::vector<BlellochScanMetadata> blelloch_metadatas_;
  rhi::ObjPtr<rhi::IBuffer> blelloch_metadata_buffer_;

  rhi::ObjPtr<rhi::IShader> gather_light_power_shader_;
  rhi::ObjPtr<rhi::IComputeProgram> gather_light_power_program_;

  rhi::ObjPtr<rhi::ICommandContext> preprocess_cmd_context_;

  rhi::ObjPtr<rhi::ISampler> linear_sampler_;
  rhi::ObjPtr<rhi::ISampler> nearest_sampler_;
  rhi::ObjPtr<rhi::IBindingGroup> render_group_, resolve_group_, light_group_, scan_group_;
};

Scene *DedicatedCast(sparkium::Scene *scene);

}  // namespace sparkium::raytracing
