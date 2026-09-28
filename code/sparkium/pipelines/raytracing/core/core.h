#pragma once
#include <map>

#include "sparkium/pipelines/raytracing/core/core_util.h"

namespace sparkium::raytracing {

class Core : public Object {
 public:
  Core(sparkium::Core &core);

  void PrepareBuiltinHitShaders();

  graphics::Core *GraphicsCore() const;

  const VirtualFileSystem &GetShadersVFS() const;

  graphics::Shader *GetShader(const std::string &name);

  graphics::ComputeProgram *GetComputeProgram(const std::string &name);

  graphics::ComputeProgram *GetGeometryLightPowerProgram(const CodeLines &geometry, const CodeLines &material);

  graphics::Image *GetImage(const std::string &name);

  graphics::Buffer *GetBuffer(const std::string &name);

 private:
  void LoadPublicShaders();

  struct GeometryLightResources {
    std::unique_ptr<graphics::Shader> shader;
    std::unique_ptr<graphics::ComputeProgram> program;
  };

  std::map<std::pair<std::string, std::string>, GeometryLightResources> geometry_light_programs_;
  bool native_shaders_ready_{false};
  sparkium::Core &core_;
};

Core *DedicatedCast(sparkium::Core *core);

void Render(sparkium::Core *core,
            sparkium::Scene *scene,
            sparkium::Camera *camera,
            sparkium::Film *film,
            bool software = false,
            bool ray_query = false);
}  // namespace sparkium::raytracing
