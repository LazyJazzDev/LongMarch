#pragma once
#include <memory>

#include "sparkium/pipelines/raster/core/core_util.h"

namespace sparkium::raster {

class Scene : public Object {
 public:
  Scene(sparkium::Scene &scene);

  bool RecordRender(sparkium::Renderer &renderer,
                    rhi::ICommandContext *commands,
                    ResourceUpdates::Cursor &updates,
                    Camera *camera,
                    Film *film);

  struct EntityStatus {
    bool active;
    bool keep;
  };

  using Settings = sparkium::Scene::Settings;
  const Settings &settings;

  std::map<Entity *, EntityStatus> entities_;

  void RegisterRenderCallback(
      const std::function<bool(sparkium::Renderer &, rhi::ICommandContext *, rhi::IBindingGroup *)> &callback);
  void RegisterLightingCallback(
      const std::function<bool(sparkium::Renderer &, rhi::ICommandContext *, Camera *camera, Film *film)> &callback);

 private:
  sparkium::Scene &scene_;
  Core *core_;

  std::vector<std::function<bool(sparkium::Renderer &, rhi::ICommandContext *, rhi::IBindingGroup *)>>
      render_callbacks_;
  std::vector<std::function<bool(sparkium::Renderer &, rhi::ICommandContext *, Camera *camera, Film *film)>>
      lighting_callbacks_;

  rhi::ObjPtr<rhi::IShader> ambient_light_vs_;
  rhi::ObjPtr<rhi::IShader> ambient_light_ps_;
  std::shared_ptr<const rhi::ProgramDesc> ambient_light_description_;
  rhi::ObjPtr<rhi::IBindingGroup> ambient_light_group_;
  rhi::ObjPtr<rhi::IBuffer> ambient_light_buffer_;
};

Scene *DedicatedCast(sparkium::Scene *scene);

}  // namespace sparkium::raster
