#include "sparkium/pipelines/raster/core/core.h"

#include "sparkium/pipelines/raster/core/camera.h"
#include "sparkium/pipelines/raster/core/film.h"
#include "sparkium/pipelines/raster/core/scene.h"

namespace sparkium::raster {

Core::Core(sparkium::Core &core) : core_(core) {
}

rhi::IDevice *Core::GetDevice() const {
  return core_.GetDevice();
}

const VirtualFileSystem &Core::GetShadersVFS() const {
  return core_.GetShadersVFS();
}

rhi::IShader *Core::GetShader(const std::string &name) {
  return core_.GetShader(name);
}

rhi::IComputeProgram *Core::GetComputeProgram(const std::string &name) {
  return core_.GetComputeProgram(name);
}

rhi::IBuffer *Core::GetBuffer(const std::string &name) {
  return core_.GetBuffer(name);
}

rhi::IImage *Core::GetImage(const std::string &name) {
  return core_.GetImage(name);
}

void Render(sparkium::Core *core, sparkium::Scene *scene, sparkium::Camera *camera, sparkium::Film *film) {
  auto raster_core = DedicatedCast(core);
  auto raster_scene = DedicatedCast(scene);
  auto raster_camera = DedicatedCast(camera);
  auto raster_film = DedicatedCast(film);
  raster_scene->Render(raster_camera, raster_film);
}

Core *DedicatedCast(sparkium::Core *core) {
  COMPONENT_CAST(core, Core)
}

}  // namespace sparkium::raster
