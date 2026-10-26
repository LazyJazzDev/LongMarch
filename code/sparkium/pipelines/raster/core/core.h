#pragma once
#include "sparkium/pipelines/raster/core/core_util.h"

namespace sparkium::raster {

class Core : public Object {
 public:
  Core(sparkium::Core &core);

  rhi::IDevice *GetDevice() const;

  const VirtualFileSystem &GetShadersVFS() const;

  rhi::IShader *GetShader(const std::string &name);

  rhi::IComputeProgram *GetComputeProgram(const std::string &name);

  rhi::IBuffer *GetBuffer(const std::string &name);

  rhi::IImage *GetImage(const std::string &name);

 private:
  sparkium::Core &core_;
};

void Render(sparkium::Core *core, sparkium::Scene *scene, sparkium::Camera *camera, sparkium::Film *film);

Core *DedicatedCast(sparkium::Core *core);

}  // namespace sparkium::raster
