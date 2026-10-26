#include "sparkium/pipelines/raytracing/entity/entity_geometry_meterial.h"

#include "sparkium/pipelines/raytracing/core/core.h"
#include "sparkium/pipelines/raytracing/core/scene.h"
#include "sparkium/pipelines/raytracing/geometry/geometries.h"
#include "sparkium/pipelines/raytracing/material/materials.h"

namespace sparkium::raytracing {

EntityGeometryMaterial::EntityGeometryMaterial(sparkium::EntityGeometryMaterial &entity)
    : entity_(entity),
      Entity(DedicatedCast(entity.GetCore())) {
  geometry_ = DedicatedCast(entity_.GetGeometry());
  material_ = DedicatedCast(entity_.GetMaterial());

  if (!geometry_ || !material_)
    throw std::runtime_error("ray tracing requires supported geometry and material components");

  light_geom_mat_ = std::make_unique<LightGeometryMaterial>(core_, geometry_, material_, entity_.transform);
}

void EntityGeometryMaterial::PrepareHitGroups() {
  if (hit_groups_.render_group.closest_hit)
    return;
  if (!core_->GetDevice()->SupportRayTracingPipeline())
    return;

  core_->PrepareBuiltinHitShaders();
  if (dynamic_cast<GeometryMesh *>(geometry_)) {
    if (dynamic_cast<MaterialLambertian *>(material_)) {
      hit_groups_.render_group.closest_hit = core_->GetShader("mesh_lambertian_chit");
      hit_groups_.shadow_group.closest_hit = core_->GetShader("mesh_lambertian_shadow_chit");
    } else if (dynamic_cast<MaterialLight *>(material_)) {
      hit_groups_.render_group.closest_hit = core_->GetShader("mesh_light_chit");
      hit_groups_.shadow_group.closest_hit = core_->GetShader("mesh_light_shadow_chit");
      hit_groups_.shadow_group.any_hit = core_->GetShader("mesh_light_shadow_ahit");
    } else if (dynamic_cast<MaterialPrincipled *>(material_)) {
      hit_groups_.render_group.closest_hit = core_->GetShader("mesh_principled_chit");
      hit_groups_.shadow_group.closest_hit = core_->GetShader("mesh_principled_shadow_chit");
    } else if (dynamic_cast<MaterialSpecular *>(material_)) {
      hit_groups_.render_group.closest_hit = core_->GetShader("mesh_specular_chit");
      hit_groups_.shadow_group.closest_hit = core_->GetShader("mesh_specular_shadow_chit");
    } else if (auto graph = dynamic_cast<MaterialShaderGraph *>(material_)) {
      graph->PrepareHitShaders();
      hit_groups_.render_group.closest_hit = graph->RenderClosestHitShader();
      hit_groups_.shadow_group.closest_hit = graph->ShadowClosestHitShader();
      hit_groups_.shadow_group.any_hit = graph->ShadowAnyHitShader();
    }
  }

  if (!hit_groups_.render_group.closest_hit) {
    auto vfs = core_->GetShadersVFS();
    vfs.WriteFile("material_sampler.slang", material_->SamplerImpl());
    vfs.WriteFile("entity_chit.slang", geometry_->ClosestHitShaderImpl());
    closest_hit_shader_ = core_->GetDevice()->CreateShader("entity_chit.slang", "RenderClosestHit",
                                                           rhi::SHADER_STAGE_CLOSEST_HIT, &vfs, {"."}, {});
    shadow_closest_hit_shader_ = core_->GetDevice()->CreateShader("entity_chit.slang", "ShadowClosestHit",
                                                                  rhi::SHADER_STAGE_CLOSEST_HIT, &vfs, {"."}, {});
    hit_groups_.render_group.closest_hit = closest_hit_shader_.Get();
    hit_groups_.shadow_group.closest_hit = shadow_closest_hit_shader_.Get();
  }
}

void EntityGeometryMaterial::Update(Scene *scene) {
  if (!geometry_ || !material_ || geometry_->PrimitiveCount() == 0)
    return;
  material_->Update(scene);
  int32_t light_index = scene->RegisterLight(light_geom_mat_.get());
  int32_t instance_index;
  if (scene->SoftwareTracing()) {
    if (!dynamic_cast<GeometryMesh *>(geometry_))
      throw std::runtime_error("compute ray tracing currently requires triangle geometry");
    instance_index = scene->RegisterSoftwareInstance(geometry_, material_, entity_.GetTransformation(), light_index);
  } else {
    PrepareHitGroups();
    instance_index = scene->RegisterInstance(
        geometry_->BLAS(), entity_.GetTransformation(), scene->RegisterHitGroup(hit_groups_),
        scene->RegisterBuffer(geometry_->Buffer()), scene->RegisterBuffer(material_->Buffer()), light_index);
  }
  scene->LightCustomIndex(light_index) = instance_index;
}

}  // namespace sparkium::raytracing
