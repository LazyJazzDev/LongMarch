#include "snow_mount/visualizer/visualizer_entity.h"

#include <snow_mount/snow_mount.h>

#include "snow_mount/visualizer/visualizer_core.h"

namespace snow_mount::visualizer {

Entity::Entity(const std::shared_ptr<Core> &core) : core_(core) {
}

int Entity::ExecuteStage(RenderStage render_stage, RenderContext *ctx) {
  return 0;
}

void Entity::PyBind(pybind11::module &m) {
  pybind11::class_<Entity, std::shared_ptr<Entity>> entity(m, "Entity");

  EntityMeshObject::PyBind(m);
}

EntityMeshObject::EntityMeshObject(const std::shared_ptr<Core> &core,
                                   const std::weak_ptr<Mesh> &mesh,
                                   const Material &material,
                                   const Matrix4<float> &transform)
    : Entity(core), mesh_(mesh) {
  program_ = core_->LoadProgram<ProgramNoNormal>(PROGRAM_ID_NO_NORMAL, []() {
    std::shared_ptr<ProgramNoNormal> program;
    return program;
  });
  info_.material = material;
  info_.model = EigenToGLM(transform);
}

int EntityMeshObject::ExecuteStage(RenderStage render_stage, RenderContext *ctx) {
  if (render_stage == RENDER_STAGE_RASTER_GEOMETRY_PASS) {
  }
  return Entity::ExecuteStage(render_stage, ctx);
}

void EntityMeshObject::PyBind(pybind11::module &m) {
  pybind11::class_<EntityMeshObject, Entity, std::shared_ptr<EntityMeshObject>> entity_mesh_object(m,
                                                                                                   "EntityMeshObject");
}

}  // namespace snow_mount::visualizer
