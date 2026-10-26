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

  void RegisterRenderCallback(const std::function<void(rhi::ICommandContext *, rhi::IBuffer *)> &callback);
  void RegisterShadowMapCallback(const std::function<void(rhi::ICommandContext *)> &callback);
  void RegisterLightingCallback(
      const std::function<void(rhi::ICommandContext *, Camera *camera, Film *film)> &callback);

 private:
  sparkium::Scene &scene_;
  Core *core_;

  std::vector<std::function<void(rhi::ICommandContext *, rhi::IBuffer *)>> render_callbacks_;
  std::vector<std::function<void(rhi::ICommandContext *)>> shadow_map_callbacks_;
  std::vector<std::function<void(rhi::ICommandContext *, Camera *camera, Film *film)>> lighting_callbacks_;

  rhi::ObjPtr<rhi::IShader> ambient_light_vs_;
  rhi::ObjPtr<rhi::IShader> ambient_light_ps_;
  rhi::ObjPtr<rhi::IProgram> ambient_light_program_;
  rhi::ObjPtr<rhi::IBindingGroup> ambient_light_group_;
  rhi::ObjPtr<rhi::IBuffer> ambient_light_buffer_;
};

Scene *DedicatedCast(sparkium::Scene *scene);

}  // namespace sparkium::raster
