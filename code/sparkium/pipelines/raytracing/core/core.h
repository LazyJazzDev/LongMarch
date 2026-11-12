#pragma once
#include <map>

#include "sparkium/pipelines/raytracing/core/core_util.h"

namespace sparkium::raytracing {

class Core : public Object {
 public:
  Core(sparkium::Core &core);

  void PrepareBuiltinHitShaders();

  rhi::IDevice *GetDevice() const;

  bool QueueUpload(rhi::IBuffer *buffer, const void *data, uint64_t size, uint64_t offset = 0) {
    return core_.QueueUpload(buffer, data, size, offset);
  }

  bool QueueBuild(rhi::IBottomLevelAccelerationStructure *structure,
                  const std::vector<rhi::AccelerationStructureGeometryBuffers> &geometries) {
    return core_.QueueBuild(structure, geometries);
  }

  bool RecordUpdates(rhi::ICommandContext *commands, ResourceUpdates::Cursor &cursor) {
    return core_.RecordUpdates(commands, cursor);
  }

  bool IsValid() const {
    return core_.IsValid();
  }

  bool ReportFailure(const char *message) {
    return core_.ReportFailure(message);
  }

  const VirtualFileSystem &GetShadersVFS() const;

  rhi::IShader *GetShader(const std::string &name);

  rhi::IComputeProgram *GetComputeProgram(const std::string &name);

  rhi::IComputeProgram *GetGeometryLightPowerProgram(const CodeLines &geometry, const CodeLines &material);

  rhi::IImage *GetImage(const std::string &name);

  rhi::IBuffer *GetBuffer(const std::string &name);

 private:
  void LoadPublicShaders();

  struct GeometryLightResources {
    rhi::ObjPtr<rhi::IShader> shader;
    rhi::ObjPtr<rhi::IComputeProgram> program;
  };

  std::map<std::pair<std::string, std::string>, GeometryLightResources> geometry_light_programs_;
  bool native_shaders_ready_{false};
  sparkium::Core &core_;
};

Core *DedicatedCast(sparkium::Core *core);

bool RecordRender(sparkium::Core *core,
                  sparkium::Renderer &renderer,
                  rhi::ICommandContext *commands,
                  ResourceUpdates::Cursor &updates,
                  sparkium::Scene *scene,
                  sparkium::Camera *camera,
                  sparkium::Film *film,
                  bool software = false,
                  bool ray_query = false);
}  // namespace sparkium::raytracing
