#pragma once
#include "sparkium/pipelines/raster/core/core_util.h"

namespace sparkium::raster {

class Scene : public Object {
 public:
  Scene(sparkium::Scene &scene);

  void Render(Camera *camera, Film *film);

  struct EntityStatus {
    bool active;
    bool keep;
  };

  using Settings = sparkium::Scene::Settings;
  const Settings &settings;

  std::map<Entity *, EntityStatus> entities_;

  void RegisterRenderCallback(const std::function<void(graphics::CommandContext *, graphics::Buffer *)> &callback);
  void RegisterShadowMapCallback(const std::function<void(graphics::CommandContext *)> &callback);
  void RegisterLightingCallback(
      const std::function<void(graphics::CommandContext *, Camera *camera, Film *film)> &callback);

 private:
  sparkium::Scene &scene_;
  Core *core_;

  std::vector<std::function<void(graphics::CommandContext *, graphics::Buffer *)>> render_callbacks_;
  std::vector<std::function<void(graphics::CommandContext *)>> shadow_map_callbacks_;
  std::vector<std::function<void(graphics::CommandContext *, Camera *camera, Film *film)>> lighting_callbacks_;

  std::unique_ptr<graphics::Shader> ambient_light_vs_;
  std::unique_ptr<graphics::Shader> ambient_light_ps_;
  std::unique_ptr<graphics::Program> ambient_light_program_;
  std::unique_ptr<graphics::Buffer> ambient_light_buffer_;
};

Scene *DedicatedCast(sparkium::Scene *scene);

}  // namespace sparkium::raster
