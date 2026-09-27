#pragma once
#include "sparkium/pipelines/raster/core/core_util.h"

namespace sparkium::raster {

class Core : public Object {
 public:
  Core(sparkium::Core &core);

  DataUpdateTracker &GetDataUpdateTracker() {
    return core_.GetDataUpdateTracker();
  }

  int CreateBuffer(size_t size, graphics::BufferType type, double_ptr<sparkium::Buffer> buffer) {
    return core_.CreateBuffer(size, type, buffer);
  }

  int CreateImage(int width, int height, graphics::ImageFormat format, double_ptr<sparkium::Image> image) {
    return core_.CreateImage(width, height, format, image);
  }

  graphics::Core *GraphicsCore() const;

  const VirtualFileSystem &GetShadersVFS() const;

  graphics::Shader *GetShader(const std::string &name);

  graphics::ComputeProgram *GetComputeProgram(const std::string &name);

  graphics::Buffer *GetBuffer(const std::string &name);

  graphics::Image *GetImage(const std::string &name);

 private:
  sparkium::Core &core_;
};

void Render(sparkium::Core *core, sparkium::Scene *scene, sparkium::Camera *camera, sparkium::Film *film);

Core *DedicatedCast(sparkium::Core *core);

}  // namespace sparkium::raster
