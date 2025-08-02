#include "sparks/entity/entity_geometry_surface.h"

#include "sparks/core/core.h"
#include "sparks/core/geometry.h"
#include "sparks/core/scene.h"
#include "sparks/core/surface.h"

namespace sparks {

EntityGeometrySurface::EntityGeometrySurface(Core *core,
                                             Geometry *geometry,
                                             Surface *surface,
                                             const glm::mat4x3 &transformation)
    : Entity(core) {
  geometry_ = geometry;
  surface_ = surface;
  transformation_ = transformation;

  CodeLines closest_hit_impl = geometry->ClosestHitShaderImpl();
  closest_hit_impl.InsertAfter(surface->SamplerImpl(), "// Surface Sampler Implementation");
  auto vfs = core_->GetShadersVFS();
  vfs.WriteFile("entity_chit.hlsl", closest_hit_impl);
  core_->GraphicsCore()->CreateShader(vfs, "entity_chit.hlsl", "EntityClosestHit", "lib_6_3", {"-I."},
                                      &closest_hit_shader_);
}

void EntityGeometrySurface::Update(Scene *scene) {
  scene->RegisterInstance(scene->RegisterGeometry(geometry_), transformation_, scene->RegisterSurface(surface_));
}

}  // namespace sparks
