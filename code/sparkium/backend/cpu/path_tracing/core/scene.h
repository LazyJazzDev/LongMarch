#pragma once
#include "sparkium/backend/cpu/path_tracing/core/core_util.h"
#include "sparkium/backend/cpu/path_tracing/core/software_pipeline.h"

namespace sparkium::cpu_tracing {

class Scene : public Object {
 public:
  Scene(sparkium::Scene &scene);

  void Render(Camera *camera, Film *film);

  bool SoftwareTracing() const {
    return true;
  }

  int32_t RegisterSoftwareInstance(Geometry *geometry,
                                   Material *material,
                                   const glm::mat4x3 &transform,
                                   int32_t custom_index);

  int32_t RegisterLight(Light *light, int custom_index = -1);

  int &LightCustomIndex(int32_t light_index);

  int &InstanceCustomIndex(int32_t instance_index);

  using Settings = sparkium::Scene::Settings;
  Settings &settings;

  struct EntityStatus {
    bool active{true};
    bool keep{false};
    int shader_version{0};
  };

  int32_t RegisterBuffer(graphics::Buffer *buffer);
  int32_t RegisterImage(graphics::Image *image);

 private:
  void UpdatePipeline(Camera *camera);
  bool rendered_{false};
  std::unique_ptr<SoftwarePipeline> software_pipeline_;
  sparkium::Scene &scene_;
  Core *core_;

  std::unique_ptr<graphics::Buffer> scene_settings_buffer_;
  std::map<Entity *, EntityStatus> entities_;

  std::vector<graphics::Buffer *> buffers_;
  std::map<graphics::Buffer *, int32_t> buffer_map_;

  std::vector<graphics::Image *> sdr_images_;
  std::map<graphics::Image *, int32_t> sdr_image_map_;

  std::vector<graphics::Image *> hdr_images_;
  std::map<graphics::Image *, int32_t> hdr_image_map_;

  bool pipeline_dirty_{true};
  int buffer_capacity_{0};
  int sdr_image_capacity_{0};
  int hdr_image_capacity_{0};

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

Scene *DedicatedCast(sparkium::Scene *scene);

}  // namespace sparkium::cpu_tracing
