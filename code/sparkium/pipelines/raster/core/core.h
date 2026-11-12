#pragma once
#include "sparkium/pipelines/raster/core/core_util.h"

namespace sparkium::raster {

class Core : public Object {
 public:
  Core(sparkium::Core &core);

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

  rhi::IBuffer *GetBuffer(const std::string &name);

  rhi::IImage *GetImage(const std::string &name);

 private:
  sparkium::Core &core_;
};

bool RecordRender(sparkium::Core *core,
                  rhi::ICommandContext *commands,
                  ResourceUpdates::Cursor &updates,
                  sparkium::Scene *scene,
                  sparkium::Camera *camera,
                  sparkium::Film *film);

Core *DedicatedCast(sparkium::Core *core);

}  // namespace sparkium::raster
