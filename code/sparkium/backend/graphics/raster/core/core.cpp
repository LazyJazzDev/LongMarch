#include "sparkium/backend/graphics/raster/core/core.h"

#include "sparkium/backend/graphics/raster/core/camera.h"
#include "sparkium/backend/graphics/raster/core/film.h"
#include "sparkium/backend/graphics/raster/core/scene.h"

namespace sparkium::raster {

Core::Core(sparkium::Core &core) : core_(core) {
}

backend::Device *Core::BackendDevice() const {
  return core_.BackendDevice();
}

const VirtualFileSystem &Core::GetShadersVFS() const {
  return core_.GetShadersVFS();
}

graphics::Shader *Core::GetShader(const std::string &name) {
  return core_.GetShader(name);
}

graphics::ComputeProgram *Core::GetComputeProgram(const std::string &name) {
  return core_.GetComputeProgram(name);
}

graphics::Buffer *Core::GetBuffer(const std::string &name) {
  return core_.GetBuffer(name);
}

graphics::Image *Core::GetImage(const std::string &name) {
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
