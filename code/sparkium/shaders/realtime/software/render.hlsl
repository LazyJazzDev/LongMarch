#define SPARKIUM_SOFTWARE_RT
#include "bindings.hlsli"
#include "bsdf/lambertian.hlsli"
#include "bsdf/principled_material.hlsli"
#include "bsdf/specular.hlsli"
#include "camera.hlsl"
#include "direct_lighting.hlsli"
#include "geometry/mesh/hit_record.hlsli"
#include "random.hlsli"
#include "software/traversal.hlsli"
#include "subsurface_random_walk.hlsli"

#ifdef SPARKIUM_SHADER_GRAPHS
#include "material/shader_graph/surface_sampler.hlsli"
#endif
#include "software_materials.hlsli"

HitRecord SoftwareHitRecord(SoftwareHit hit, float3 direction) {
  SoftwareInstance instance = LoadSoftwareInstance(software_instances, hit.instance);
  return MakeMeshHitRecord(instance.geometry, hit.instance, hit.primitive, hit.barycentric, hit.distance, direction,
                           instance.object_to_world, transpose(instance.world_to_object));
}

void ApplyPathMiss(inout RenderContext context);

void SoftwareTracePath(RayDesc ray, inout RenderContext context) {
  SoftwareHit hit;
  if (!InlineIntersect(ray, false, hit)) {
    ApplyPathMiss(context);
    return;
  }

  HitRecord record = SoftwareHitRecord(hit, ray.Direction);
  if (!ContinueSubsurfaceRandomWalk(context, record))
    SoftwareSampleMaterial(LoadSoftwareInstance(software_instances, hit.instance).material, context, record);
}

void ApplyPathMiss(inout RenderContext context) {
  if (context.medium_object_index < 0)
    context.radiance += render_settings.background_color * context.throughput;
  context.throughput = float3(0, 0, 0);
}

#include "software/shadow.hlsli"
